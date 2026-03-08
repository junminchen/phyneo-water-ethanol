#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
from jax import jit, value_and_grad
from openmm import app
from openmm.unit import nanometer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MD_ROOT = SCRIPT_DIR.parents[0]
BUNDLED_DMFF_ROOT = MD_ROOT / "vendor" / "DMFF"
LOCAL_DMFF_ROOT = REPO_ROOT / "DMFF"
for dmff_root in (BUNDLED_DMFF_ROOT, LOCAL_DMFF_ROOT):
    if dmff_root.exists() and str(dmff_root) not in sys.path:
        sys.path.insert(0, str(dmff_root))

from dmff.api import Hamiltonian
from dmff.common import nblist


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-point DMFF energy check for the CMD_H2O water model."
    )
    parser.add_argument("--pdb", default=str(SCRIPT_DIR / "02molL_init.pdb"))
    parser.add_argument(
        "--ff-xml",
        default=str(SCRIPT_DIR / "ff.backend_amoeba_total1000_classical_intra.xml"),
    )
    parser.add_argument("--residues-xml", default=str(SCRIPT_DIR / "residues.xml"))
    parser.add_argument("--cutoff-nm", type=float, default=0.6)
    parser.add_argument("--pol-steps", type=int, default=20)
    return parser.parse_args()


def build_energy_terms(
    pdb_path: Path,
    ff_xml: Path,
    residues_xml: Path,
    cutoff_nm: float,
    pol_steps: int | None = None,
):
    app.Topology.loadBondDefinitions(str(residues_xml))
    mol = app.PDBFile(str(pdb_path))
    pos = jnp.array(mol.positions._value)
    box = jnp.array(mol.topology.getPeriodicBoxVectors()._value)
    box_length = float(box[0][0])

    hamiltonian = Hamiltonian(str(ff_xml))
    potential_kwargs = {
        "nonbondedCutoff": cutoff_nm * nanometer,
        "nonbondedMethod": app.PME,
        "ethresh": 1e-4,
    }
    if pol_steps is not None:
        potential_kwargs["step_pol"] = pol_steps
    potentials = hamiltonian.createPotential(mol.topology, **potential_kwargs)
    efunc = potentials.getPotentialFunc()
    params = hamiltonian.getParameters()

    nbl = nblist.NeighborListFreud(box, cutoff_nm, potentials.meta["cov_map"], padding=False)
    nbl.allocate(pos, box)
    pairs = nbl.pairs

    def total_energy_fn(pos_nm, box_len_nm, pair_index):
        box_nm = jnp.array(
            [[box_len_nm, 0.0, 0.0], [0.0, box_len_nm, 0.0], [0.0, 0.0, box_len_nm]]
        )
        return efunc(pos_nm, box_nm, pair_index, params)

    return mol, pos, box_length, pairs, jit(value_and_grad(total_energy_fn, argnums=(0, 1)))


def main():
    args = parse_args()
    pdb_path = Path(args.pdb).resolve()
    ff_xml = Path(args.ff_xml).resolve()
    residues_xml = Path(args.residues_xml).resolve()

    mol, pos, box_length, pairs, total_fn = build_energy_terms(
        pdb_path,
        ff_xml,
        residues_xml,
        args.cutoff_nm,
        args.pol_steps,
    )
    energy_total, (grad, virial) = total_fn(pos, box_length, pairs)

    atom_count = sum(1 for _ in mol.topology.atoms())
    residue_count = sum(1 for _ in mol.topology.residues())

    print(f"pdb={pdb_path}")
    print(f"ff_xml={ff_xml}")
    print(f"atoms={atom_count} residues={residue_count} pairs={pairs.shape[0]}")
    print(f"polarization_solver={'fixed_'+str(args.pol_steps) if args.pol_steps is not None else 'converged'}")
    print(f"box_length_nm={box_length:.6f}")
    print(f"energy_total_kjmol={float(energy_total):.12f}")
    print(f"grad_l2_kjmol_per_nm={float(jnp.linalg.norm(grad)):.12f}")
    print(f"virial_raw={jnp.asarray(virial)}")


if __name__ == "__main__":
    main()
