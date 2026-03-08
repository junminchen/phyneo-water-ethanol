#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from openmm.app import CutoffPeriodic, PDBFile
from openmm.unit import angstrom

from train_dimer_backend import REPO_ROOT, ensure_dmff

Hamiltonian, _, nblist = ensure_dmff(REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild local long-range and short-range dimer datasets.")
    parser.add_argument("--ff", type=Path, default=REPO_ROOT / "ff.xml")
    parser.add_argument("--input-data", type=Path, default=REPO_ROOT / "data.pickle")
    parser.add_argument("--output-sr", type=Path, required=True)
    parser.add_argument("--output-lr", type=Path, required=True)
    parser.add_argument("--step-pol", type=int, default=5)
    parser.add_argument("--report-every", type=int, default=1)
    return parser.parse_args()


def canonical_pair_name(pair_name: str) -> str:
    return pair_name[:-3] if pair_name.endswith("_sr") else pair_name


def pair_to_files(pair_name: str) -> tuple[Path, Path, Path]:
    _, monomer_a, monomer_b = canonical_pair_name(pair_name).split("_")
    dimer = REPO_ROOT / "dimer" / f"{monomer_a}_{monomer_b}_dimer.pdb"
    pdb_a = REPO_ROOT / "monomer" / f"{monomer_a}.pdb"
    pdb_b = REPO_ROOT / "monomer" / f"{monomer_b}.pdb"
    return dimer, pdb_a, pdb_b


class LongRangeKernel:
    def __init__(self, ff_path: Path, pair_name: str, step_pol: int):
        dimer_pdb, pdb_a, pdb_b = pair_to_files(pair_name)
        pdb = PDBFile(str(dimer_pdb))
        pdb_A = PDBFile(str(pdb_a))
        pdb_B = PDBFile(str(pdb_b))

        self.H = Hamiltonian(str(ff_path))
        self.H_A = Hamiltonian(str(ff_path))
        self.H_B = Hamiltonian(str(ff_path))

        self.pots = self.H.createPotential(
            pdb.topology,
            nonbondedCutoff=15 * angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4,
            step_pol=step_pol,
        )
        self.pots_A = self.H_A.createPotential(
            pdb_A.topology,
            nonbondedCutoff=15 * angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4,
            step_pol=step_pol,
        )
        self.pots_B = self.H_B.createPotential(
            pdb_B.topology,
            nonbondedCutoff=15 * angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4,
            step_pol=step_pol,
        )

        self.generators = self.H.getGenerators()
        self.generators_A = self.H_A.getGenerators()
        self.generators_B = self.H_B.getGenerators()

        self.pos = jnp.array(pdb.positions._value) * 10.0
        self.pos_A = jnp.array(pdb_A.positions._value) * 10.0
        self.pos_B = jnp.array(pdb_B.positions._value) * 10.0
        self.box = jnp.array(pdb.topology.getPeriodicBoxVectors()._value) * 10.0
        self.rc = 14.0

        self.nblist = nblist.NeighborList(self.box, self.rc, self.pots.meta["cov_map"])
        self.nblist_A = nblist.NeighborList(self.box, self.rc, self.pots_A.meta["cov_map"])
        self.nblist_B = nblist.NeighborList(self.box, self.rc, self.pots_B.meta["cov_map"])
        self.nblist.allocate(self.pos)
        self.nblist_A.allocate(self.pos_A)
        self.nblist_B.allocate(self.pos_B)

        self.pairs_AB = self.nblist.pairs[self.nblist.pairs[:, 0] < self.nblist.pairs[:, 1]]
        self.pairs_A = self.nblist_A.pairs[self.nblist_A.pairs[:, 0] < self.nblist_A.pairs[:, 1]]
        self.pairs_B = self.nblist_B.pairs[self.nblist_B.pairs[:, 0] < self.nblist_B.pairs[:, 1]]

        self.pots_es = self.pots.dmff_potentials["ADMPPmeForce"]
        self.pots_es_A = self.pots_A.dmff_potentials["ADMPPmeForce"]
        self.pots_es_B = self.pots_B.dmff_potentials["ADMPPmeForce"]
        self.pots_disp = self.pots.dmff_potentials["ADMPDispPmeForce"]
        self.pots_disp_A = self.pots_A.dmff_potentials["ADMPDispPmeForce"]
        self.pots_disp_B = self.pots_B.dmff_potentials["ADMPDispPmeForce"]

        self.predict = jax.jit(jax.vmap(self.cal_E, in_axes=(None, 0, 0), out_axes=(0, 0, 0)))

    def cal_E(self, params, pos_A, pos_B):
        pos_A = pos_A * 0.1
        pos_B = pos_B * 0.1
        pos_AB = jnp.concatenate([pos_A, pos_B], axis=0)
        box = self.box

        E_espol_A = self.pots_es_A(pos_A, box, self.pairs_A, params)
        E_espol_B = self.pots_es_B(pos_B, box, self.pairs_B, params)
        E_espol = self.pots_es(pos_AB, box, self.pairs_AB, params) - E_espol_A - E_espol_B

        pme_generator_AB = self.generators[0]
        pme_generator_A = self.generators_A[0]
        pme_generator_B = self.generators_B[0]
        U_ind_AB = jnp.vstack((pme_generator_A.pme_force.U_ind, pme_generator_B.pme_force.U_ind))
        params_pme = params["ADMPPmeForce"]
        map_atypes = self.pots.meta["ADMPPmeForce_map_atomtype"]
        map_poltypes = self.pots.meta["ADMPPmeForce_map_poltype"]
        Q_local = params_pme["Q_local"][map_atypes]
        pol = params_pme["pol"][map_poltypes]
        tholes = params_pme["thole"][map_poltypes]
        pme_force = pme_generator_AB.pme_force
        E_nonpol_AB = pme_force.energy_fn(
            pos_AB * 10.0,
            box * 10.0,
            self.pairs_AB,
            Q_local,
            U_ind_AB,
            pol,
            tholes,
            pme_generator_AB.mScales,
            pme_generator_AB.pScales,
            pme_generator_AB.dScales,
        )
        E_lr_es = E_nonpol_AB - E_espol_A - E_espol_B
        E_lr_pol = E_espol - E_lr_es
        E_lr_disp = (
            self.pots_disp(pos_AB, box, self.pairs_AB, params)
            - self.pots_disp_A(pos_A, box, self.pairs_A, params)
            - self.pots_disp_B(pos_B, box, self.pairs_B, params)
        )
        return E_lr_es, E_lr_pol, E_lr_disp


def main() -> None:
    args = parse_args()
    with open(args.input_data, "rb") as ifile:
        raw_data = pickle.load(ifile)

    params = Hamiltonian(str(args.ff)).getParameters()
    kernels = {
        pair_name: LongRangeKernel(args.ff, pair_name, args.step_pol)
        for pair_name in raw_data.keys()
    }

    data_sr = copy.deepcopy(raw_data)
    data_lr = {}

    for pair_name in raw_data:
        data_lr[pair_name] = {}
        kernel = kernels[pair_name]
        for index, sid in enumerate(sorted(raw_data[pair_name].keys())):
            scan_res = data_sr[pair_name][sid]
            scan_res.pop("shift", None)
            scan_res["tot_full"] = np.asarray(scan_res["tot"]).copy()

            E_es, E_pol, E_disp = kernel.predict(params, scan_res["posA"], scan_res["posB"])
            E_es = np.asarray(E_es)
            E_pol = np.asarray(E_pol)
            E_disp = np.asarray(E_disp)

            scan_res["es"] = np.asarray(scan_res["es"]) - E_es
            scan_res["pol"] = np.asarray(scan_res["pol"]) - E_pol
            scan_res["disp"] = np.asarray(scan_res["disp"]) - E_disp
            scan_res["tot"] = np.asarray(scan_res["tot"]) - (E_es + E_pol + E_disp)

            data_lr[pair_name][sid] = {
                "es": E_es,
                "pol": E_pol,
                "disp": E_disp,
                "tot": E_es + E_pol + E_disp,
            }
            if index % max(args.report_every, 1) == 0:
                print(f"{pair_name} {sid}")

    args.output_sr.parent.mkdir(parents=True, exist_ok=True)
    args.output_lr.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_sr, "wb") as ofile:
        pickle.dump(data_sr, ofile)
    with open(args.output_lr, "wb") as ofile:
        pickle.dump(data_lr, ofile)

    print(f"data_sr: {args.output_sr}")
    print(f"data_lr: {args.output_lr}")


if __name__ == "__main__":
    main()
