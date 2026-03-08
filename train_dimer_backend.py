#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import os
import pickle
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp
import numpy as np
from openmm.app import CutoffPeriodic, PDBFile
from openmm.unit import angstrom

try:
    import optax
except ModuleNotFoundError:
    optax = None

COMPONENTS = ("ex", "es", "pol", "disp", "dhf", "tot")
DEFAULT_COMPONENT_WEIGHTS = {
    "ex": 0.1,
    "es": 0.1,
    "pol": 0.1,
    "disp": 0.1,
    "dhf": 0.1,
    "tot": 1.0,
}
DEFAULT_TRAINABLE = ("A_ex", "A_es", "A_pol", "A_disp", "A_dhf", "B")
DEFAULT_PAIRS = ("Pairs_EA_EA", "Pairs_EA_H2O", "Pairs_H2O_H2O")


def ensure_dmff(repo_root: Path):
    if "jax.config" not in sys.modules:
        shim = types.ModuleType("jax.config")
        shim.config = jax.config
        sys.modules["jax.config"] = shim

    candidate_roots = (
        repo_root / "DMFF",
        repo_root / "3_MD" / "vendor" / "DMFF",
    )
    for local_dmff in candidate_roots:
        if local_dmff.exists() and str(local_dmff) not in sys.path:
            sys.path.insert(0, str(local_dmff))
            break

    from dmff.api import Hamiltonian  # type: ignore
    from dmff.api.paramset import ParamSet  # type: ignore
    from dmff.common import nblist  # type: ignore

    return Hamiltonian, ParamSet, nblist


REPO_ROOT = Path(__file__).resolve().parent
Hamiltonian, ParamSet, nblist = ensure_dmff(REPO_ROOT)


@dataclass
class AdamState:
    step: jnp.ndarray
    m: object
    v: object


class SimpleAdam:
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def init(self, params) -> AdamState:
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return AdamState(step=jnp.array(0, dtype=jnp.int32), m=zeros, v=zeros)

    def update(self, grads, state: AdamState, params):
        del params
        step = state.step + 1
        m = jax.tree_util.tree_map(lambda old_m, grad: self.beta1 * old_m + (1.0 - self.beta1) * grad, state.m, grads)
        v = jax.tree_util.tree_map(lambda old_v, grad: self.beta2 * old_v + (1.0 - self.beta2) * (grad**2), state.v, grads)
        step_f = step.astype(jnp.float64)
        m_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - self.beta1**step_f), m)
        v_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - self.beta2**step_f), v)
        updates = jax.tree_util.tree_map(
            lambda m_i, v_i: -self.learning_rate * m_i / (jnp.sqrt(v_i) + self.eps),
            m_hat,
            v_hat,
        )
        return updates, AdamState(step=step, m=m, v=v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faithful backend reproduction of the original dimer training notebook.")
    parser.add_argument("--ff", type=Path, default=REPO_ROOT / "ff.xml")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "2_PES" / "data_sr.pickle")
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260307)
    parser.add_argument("--cutoff-angstrom", type=float, default=25.0)
    parser.add_argument("--box-angstrom", type=float, default=60.0)
    parser.add_argument("--weight-threshold", type=float, default=20.0)
    parser.add_argument("--weight-kt", type=float, default=2.494)
    parser.add_argument("--random-a-scale", type=float, default=100.0)
    parser.add_argument("--trainable", nargs="+", default=list(DEFAULT_TRAINABLE))
    parser.add_argument(
        "--component-weights",
        nargs="+",
        default=[f"{name}={value}" for name, value in DEFAULT_COMPONENT_WEIGHTS.items()],
    )
    parser.add_argument("--optimizer", choices=("optax", "simple_adam"), default="optax")
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "params" / "backend_faithful")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument("--restart", type=Path, default=None)
    parser.add_argument("--write-xml", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1, help="Must stay 1 to match the original notebook.")
    return parser.parse_args()


def parse_component_weights(entries: Sequence[str]) -> Dict[str, float]:
    weights = dict(DEFAULT_COMPONENT_WEIGHTS)
    for entry in entries:
        key, value = entry.split("=", 1)
        if key not in COMPONENTS:
            raise ValueError(f"Unknown component key: {key}")
        weights[key] = float(value)
    return weights


def pair_to_files(pair_name: str) -> tuple[Path, Path, Path]:
    _, monomer_a, monomer_b = pair_name.split("_")
    dimer = REPO_ROOT / "dimer" / f"{monomer_a}_{monomer_b}_dimer.pdb"
    pdb_a = REPO_ROOT / "monomer" / f"{monomer_a}.pdb"
    pdb_b = REPO_ROOT / "monomer" / f"{monomer_b}.pdb"
    return dimer, pdb_a, pdb_b


def calculate_weights(energy_total_full: np.ndarray, threshold: float, kT: float) -> np.ndarray:
    weights = np.where(energy_total_full < threshold, 1.0, np.exp(-(energy_total_full - threshold) / kT))
    return weights.astype(np.float64)


def load_training_data(args: argparse.Namespace):
    with open(args.data, "rb") as ifile:
        raw_data = pickle.load(ifile)

    data = {}
    for pair_name in args.pairs:
        pair_data = {}
        for sid, scan in raw_data[pair_name].items():
            item = {}
            for key, value in scan.items():
                item[key] = np.asarray(value, dtype=np.float64)
            item["wts"] = calculate_weights(item["tot_full"], args.weight_threshold, args.weight_kt)
            pair_data[sid] = {key: jnp.asarray(value) for key, value in item.items()}
        data[pair_name] = pair_data
    return data


def init_shared_params(params0: ParamSet, restart: Path | None, random_a_scale: float, seed: int):
    if restart is not None:
        with open(restart, "rb") as ifile:
            loaded = pickle.load(ifile)
        if isinstance(loaded, ParamSet):
            loaded = loaded.parameters
        return loaded

    base = params0.parameters
    params = {}
    sr_forces = {
        "ex": "SlaterExForce",
        "es": "SlaterSrEsForce",
        "pol": "SlaterSrPolForce",
        "disp": "SlaterSrDispForce",
        "dhf": "SlaterDhfForce",
    }

    for key, value in base["ADMPPmeForce"].items():
        params[key] = jnp.array(value)
    for key, value in base["ADMPDispPmeForce"].items():
        params[key] = jnp.array(value)

    for component, force_name in sr_forces.items():
        for key, value in base[force_name].items():
            if key == "A":
                params[f"A_{component}"] = jnp.array(value)
            else:
                params[key] = jnp.array(value)

    rng = np.random.default_rng(seed)
    for component in sr_forces:
        params[f"A_{component}"] = jnp.asarray(rng.random(params[f"A_{component}"].shape) * random_a_scale, dtype=jnp.float64)

    params["Q"] = jnp.array(base["QqTtDampingForce"]["Q"])
    return params


def params_convert(params):
    params_ex = {}
    params_sr_es = {}
    params_sr_pol = {}
    params_sr_disp = {}
    params_dhf = {}
    params_dmp_es = {}
    params_dmp_disp = {}

    for key in ("B",):
        params_ex[key] = params[key]
        params_sr_es[key] = params[key]
        params_sr_pol[key] = params[key]
        params_sr_disp[key] = params[key]
        params_dhf[key] = params[key]
        params_dmp_es[key] = params[key]
        params_dmp_disp[key] = params[key]

    if "C" in params:
        params_ex["C"] = params["C"]
    if "D" in params:
        params_ex["D"] = params["D"]

    params_ex["A"] = params["A_ex"]
    params_sr_es["A"] = params["A_es"]
    params_sr_pol["A"] = params["A_pol"]
    params_sr_disp["A"] = params["A_disp"]
    params_dhf["A"] = params["A_dhf"]

    params_dmp_es["Q"] = params["Q"]
    params_dmp_disp["C6"] = params["C6"]
    params_dmp_disp["C8"] = params["C8"]
    params_dmp_disp["C10"] = params["C10"]

    return {
        "SlaterExForce": params_ex,
        "SlaterSrEsForce": params_sr_es,
        "SlaterSrPolForce": params_sr_pol,
        "SlaterSrDispForce": params_sr_disp,
        "SlaterDhfForce": params_dhf,
        "QqTtDampingForce": params_dmp_es,
        "SlaterDampingForce": params_dmp_disp,
    }


def merge_shared_params_into_paramset(base_paramset: ParamSet, shared_params):
    merged = copy.deepcopy(base_paramset.parameters)
    merged["ADMPDispPmeForce"]["C6"] = shared_params["C6"]
    merged["ADMPDispPmeForce"]["C8"] = shared_params["C8"]
    merged["ADMPDispPmeForce"]["C10"] = shared_params["C10"]

    merged["SlaterExForce"]["A"] = shared_params["A_ex"]
    merged["SlaterExForce"]["B"] = shared_params["B"]
    if "C" in merged["SlaterExForce"] and "C" in shared_params:
        merged["SlaterExForce"]["C"] = shared_params["C"]
    if "D" in merged["SlaterExForce"] and "D" in shared_params:
        merged["SlaterExForce"]["D"] = shared_params["D"]

    merged["SlaterSrEsForce"]["A"] = shared_params["A_es"]
    merged["SlaterSrEsForce"]["B"] = shared_params["B"]
    merged["SlaterSrPolForce"]["A"] = shared_params["A_pol"]
    merged["SlaterSrPolForce"]["B"] = shared_params["B"]
    merged["SlaterSrDispForce"]["A"] = shared_params["A_disp"]
    merged["SlaterSrDispForce"]["B"] = shared_params["B"]
    merged["SlaterDhfForce"]["A"] = shared_params["A_dhf"]
    merged["SlaterDhfForce"]["B"] = shared_params["B"]

    merged["QqTtDampingForce"]["B"] = shared_params["B"]
    merged["QqTtDampingForce"]["Q"] = shared_params["Q"]
    merged["SlaterDampingForce"]["B"] = shared_params["B"]
    merged["SlaterDampingForce"]["C6"] = shared_params["C6"]
    merged["SlaterDampingForce"]["C8"] = shared_params["C8"]
    merged["SlaterDampingForce"]["C10"] = shared_params["C10"]

    return ParamSet(data=merged, mask=copy.deepcopy(base_paramset.mask))


class PairKernel:
    def __init__(self, ff_path: Path, pair_name: str, cutoff_angstrom: float, box_angstrom: float):
        dimer_pdb, pdb_a, pdb_b = pair_to_files(pair_name)
        self.pair_name = pair_name
        self.hamiltonian = Hamiltonian(str(ff_path))

        pdb = PDBFile(str(dimer_pdb))
        pdb_A = PDBFile(str(pdb_a))
        pdb_B = PDBFile(str(pdb_b))

        self.pots = self.hamiltonian.createPotential(
            pdb.topology,
            nonbondedCutoff=cutoff_angstrom * angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4,
        )
        self.pots_A = self.hamiltonian.createPotential(
            pdb_A.topology,
            nonbondedCutoff=cutoff_angstrom * angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4,
        )
        self.pots_B = self.hamiltonian.createPotential(
            pdb_B.topology,
            nonbondedCutoff=cutoff_angstrom * angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4,
        )

        self.pos = jnp.array(pdb.positions._value)
        self.pos_A = jnp.array(pdb_A.positions._value)
        self.pos_B = jnp.array(pdb_B.positions._value)

        self.box = jnp.eye(3, dtype=jnp.float64) * (box_angstrom * 0.1)
        self.rc = cutoff_angstrom * 0.1

        self.nblist = nblist.NeighborList(self.box, self.rc, self.pots.meta["cov_map"])
        self.nblist_A = nblist.NeighborList(self.box, self.rc, self.pots_A.meta["cov_map"])
        self.nblist_B = nblist.NeighborList(self.box, self.rc, self.pots_B.meta["cov_map"])
        self.nblist.allocate(self.pos)
        self.nblist_A.allocate(self.pos_A)
        self.nblist_B.allocate(self.pos_B)

        self.pairs_AB = self.nblist.pairs[self.nblist.pairs[:, 0] < self.nblist.pairs[:, 1]]
        self.pairs_A = self.nblist_A.pairs[self.nblist_A.pairs[:, 0] < self.nblist_A.pairs[:, 1]]
        self.pairs_B = self.nblist_B.pairs[self.nblist_B.pairs[:, 0] < self.nblist_B.pairs[:, 1]]

        self.pots_ex = self.pots.dmff_potentials["SlaterExForce"]
        self.pots_ex_A = self.pots_A.dmff_potentials["SlaterExForce"]
        self.pots_ex_B = self.pots_B.dmff_potentials["SlaterExForce"]

        self.pots_sr_es = self.pots.dmff_potentials["SlaterSrEsForce"]
        self.pots_sr_es_A = self.pots_A.dmff_potentials["SlaterSrEsForce"]
        self.pots_sr_es_B = self.pots_B.dmff_potentials["SlaterSrEsForce"]

        self.pots_sr_pol = self.pots.dmff_potentials["SlaterSrPolForce"]
        self.pots_sr_pol_A = self.pots_A.dmff_potentials["SlaterSrPolForce"]
        self.pots_sr_pol_B = self.pots_B.dmff_potentials["SlaterSrPolForce"]

        self.pots_sr_disp = self.pots.dmff_potentials["SlaterSrDispForce"]
        self.pots_sr_disp_A = self.pots_A.dmff_potentials["SlaterSrDispForce"]
        self.pots_sr_disp_B = self.pots_B.dmff_potentials["SlaterSrDispForce"]

        self.pots_dhf = self.pots.dmff_potentials["SlaterDhfForce"]
        self.pots_dhf_A = self.pots_A.dmff_potentials["SlaterDhfForce"]
        self.pots_dhf_B = self.pots_B.dmff_potentials["SlaterDhfForce"]

        self.pots_dmp_es = self.pots.dmff_potentials["QqTtDampingForce"]
        self.pots_dmp_es_A = self.pots_A.dmff_potentials["QqTtDampingForce"]
        self.pots_dmp_es_B = self.pots_B.dmff_potentials["QqTtDampingForce"]

        self.pots_dmp_disp = self.pots.dmff_potentials["SlaterDampingForce"]
        self.pots_dmp_disp_A = self.pots_A.dmff_potentials["SlaterDampingForce"]
        self.pots_dmp_disp_B = self.pots_B.dmff_potentials["SlaterDampingForce"]

        self.predict_scan = jax.jit(
            jax.vmap(self.cal_E, in_axes=(None, 0, 0), out_axes=(0, 0, 0, 0, 0, 0))
        )

    def cal_E(self, shared_params, pos_A_angstrom: jnp.ndarray, pos_B_angstrom: jnp.ndarray):
        params = params_convert(shared_params)
        pos_A = pos_A_angstrom * 0.1
        pos_B = pos_B_angstrom * 0.1
        pos_AB = jnp.concatenate([pos_A, pos_B], axis=0)

        E_ex = (
            self.pots_ex(pos_AB, self.box, self.pairs_AB, params)
            - self.pots_ex_A(pos_A, self.box, self.pairs_A, params)
            - self.pots_ex_B(pos_B, self.box, self.pairs_B, params)
        )
        E_dmp_es = (
            self.pots_dmp_es(pos_AB, self.box, self.pairs_AB, params)
            - self.pots_dmp_es_A(pos_A, self.box, self.pairs_A, params)
            - self.pots_dmp_es_B(pos_B, self.box, self.pairs_B, params)
        )
        E_sr_es = (
            self.pots_sr_es(pos_AB, self.box, self.pairs_AB, params)
            - self.pots_sr_es_A(pos_A, self.box, self.pairs_A, params)
            - self.pots_sr_es_B(pos_B, self.box, self.pairs_B, params)
        )
        E_sr_pol = (
            self.pots_sr_pol(pos_AB, self.box, self.pairs_AB, params)
            - self.pots_sr_pol_A(pos_A, self.box, self.pairs_A, params)
            - self.pots_sr_pol_B(pos_B, self.box, self.pairs_B, params)
        )
        E_dmp_disp = (
            self.pots_dmp_disp(pos_AB, self.box, self.pairs_AB, params)
            - self.pots_dmp_disp_A(pos_A, self.box, self.pairs_A, params)
            - self.pots_dmp_disp_B(pos_B, self.box, self.pairs_B, params)
        )
        E_sr_disp = (
            self.pots_sr_disp(pos_AB, self.box, self.pairs_AB, params)
            - self.pots_sr_disp_A(pos_A, self.box, self.pairs_A, params)
            - self.pots_sr_disp_B(pos_B, self.box, self.pairs_B, params)
        )
        E_dhf = (
            self.pots_dhf(pos_AB, self.box, self.pairs_AB, params)
            - self.pots_dhf_A(pos_A, self.box, self.pairs_A, params)
            - self.pots_dhf_B(pos_B, self.box, self.pairs_B, params)
        )

        E_es = E_dmp_es + E_sr_es
        E_pol = E_sr_pol
        E_disp = E_dmp_disp + E_sr_disp
        E_tot = E_ex + E_es + E_pol + E_disp + E_dhf
        return E_ex, E_es, E_pol, E_disp, E_dhf, E_tot


def make_loss_and_grad(pair_kernel: PairKernel, component_weights: Dict[str, float]):
    weights_comps = jnp.asarray([component_weights[name] for name in COMPONENTS], dtype=jnp.float64)

    def mse_loss(shared_params, scan_res):
        E_ex, E_es, E_pol, E_disp, E_dhf, E_tot = pair_kernel.predict_scan(shared_params, scan_res["posA"], scan_res["posB"])
        energies = {
            "ex": E_ex,
            "es": E_es,
            "pol": E_pol,
            "disp": E_disp,
            "dhf": E_dhf,
            "tot": E_tot,
        }
        weights_pts = scan_res["wts"]
        norm = weights_pts / jnp.sum(weights_pts)
        errs = []
        for component in COMPONENTS:
            diff = scan_res[component] - energies[component]
            errs.append(jnp.sum(diff**2 * norm))
        return jnp.sum(weights_comps * jnp.stack(errs))

    return jax.jit(jax.value_and_grad(mse_loss))


def mask_gradients(grads, trainable_keys: Iterable[str]):
    masked = copy.deepcopy(grads)
    trainable = set(trainable_keys)
    for key in list(masked.keys()):
        if key not in trainable:
            masked[key] = jnp.zeros_like(masked[key])
    return masked


def clip_gradients(grads, max_norm: float):
    leaves = jax.tree_util.tree_leaves(grads)
    global_norm = jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))
    scale = jnp.minimum(1.0, max_norm / (global_norm + 1e-12))
    return jax.tree_util.tree_map(lambda grad: grad * scale, grads)


def apply_updates(params, updates):
    return jax.tree_util.tree_map(lambda param, update: param + update, params, updates)


def save_checkpoint(params, base_paramset: ParamSet, save_dir: Path, epoch: int, ff_path: Path, write_xml: bool):
    save_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = save_dir / f"params.epoch{epoch:04d}.pickle"
    latest_path = save_dir / "latest.pickle"
    with open(pickle_path, "wb") as ofile:
        pickle.dump(params, ofile)
    with open(latest_path, "wb") as ofile:
        pickle.dump(params, ofile)

    if write_xml:
        hamiltonian = Hamiltonian(str(ff_path))
        hamiltonian.paramset = merge_shared_params_into_paramset(base_paramset, params)
        hamiltonian.renderXML(str(save_dir / f"params.epoch{epoch:04d}.xml"))
        hamiltonian.renderXML(str(save_dir / "latest.xml"))


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("`--batch-size` must stay 1 to faithfully match the original notebook training loop.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    component_weights = parse_component_weights(args.component_weights)

    data = load_training_data(args)
    kernels = {
        pair_name: PairKernel(args.ff, pair_name, args.cutoff_angstrom, args.box_angstrom)
        for pair_name in args.pairs
    }
    loss_and_grad = {
        pair_name: make_loss_and_grad(kernels[pair_name], component_weights)
        for pair_name in args.pairs
    }

    base_paramset = kernels[args.pairs[0]].hamiltonian.getParameters()
    params = init_shared_params(base_paramset, args.restart, args.random_a_scale, args.seed)

    if args.dry_run:
        pair_name = args.pairs[0]
        sid = next(iter(data[pair_name]))
        loss, grads = loss_and_grad[pair_name](params, data[pair_name][sid])
        grads = mask_gradients(grads, args.trainable)
        print(f"dry-run pair={pair_name} sid={sid} loss={float(loss):.6f}")
        print("param_keys:", " ".join(sorted(params.keys())))
        print("grad_keys:", " ".join(sorted(grads.keys())))
        return

    if args.optimizer == "optax":
        if optax is None:
            raise RuntimeError("optax is not available in this environment.")
        optimizer = optax.adam(args.lr)
    else:
        optimizer = SimpleAdam(args.lr)
    opt_state = optimizer.init(params)

    trunk = [(pair_name, sid) for pair_name in args.pairs for sid in data[pair_name]]
    print("pairs:", ", ".join(f"{pair}={len(data[pair])}" for pair in args.pairs), flush=True)
    print("trainable:", " ".join(args.trainable), flush=True)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(trunk)
        epoch_losses = []
        for pair_name, sid in trunk:
            loss, grads = loss_and_grad[pair_name](params, data[pair_name][sid])
            grads = mask_gradients(grads, args.trainable)
            if args.optimizer == "simple_adam":
                grads = clip_gradients(grads, 10.0)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            if args.optimizer == "optax":
                params = optax.apply_updates(params, updates)
            else:
                params = apply_updates(params, updates)
            epoch_losses.append(float(loss))

        mean_loss = float(np.mean(epoch_losses))
        if epoch % args.report_every == 0 or epoch == 1 or epoch == args.epochs:
            print(f"epoch={epoch:04d} mean_loss={mean_loss:.6f}", flush=True)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(params, base_paramset, args.save_dir, epoch, args.ff, args.write_xml)


if __name__ == "__main__":
    main()
