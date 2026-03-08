#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
from openmm.unit import AVOGADRO_CONSTANT_NA, joule, kilojoule_per_mole

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MD_ROOT = SCRIPT_DIR.parents[0]
BUNDLED_DMFF_ROOT = MD_ROOT / "vendor" / "DMFF"
LOCAL_DMFF_ROOT = REPO_ROOT / "DMFF"
for dmff_root in (BUNDLED_DMFF_ROOT, LOCAL_DMFF_ROOT):
    if dmff_root.exists() and str(dmff_root) not in sys.path:
        sys.path.insert(0, str(dmff_root))

from test_dmff import build_energy_terms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the energy/gradient path used by test_dmff.py and client_dmff.py."
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


def main():
    args = parse_args()
    pdb_path = Path(args.pdb).resolve()
    ff_xml = Path(args.ff_xml).resolve()
    residues_xml = Path(args.residues_xml).resolve()

    mol_ref, pos_ref, box_length_ref, pairs_ref, ref_total_fn = build_energy_terms(
        pdb_path,
        ff_xml,
        residues_xml,
        args.cutoff_nm,
        args.pol_steps,
    )
    energy_ref, (grad_ref, virial_ref) = ref_total_fn(pos_ref, box_length_ref, pairs_ref)

    mol_cli, pos_cli, box_length_cli, pairs_cli, cli_total_fn = build_energy_terms(
        pdb_path,
        ff_xml,
        residues_xml,
        args.cutoff_nm,
        args.pol_steps,
    )
    energy_cli, (grad_cli, virial_cli) = cli_total_fn(pos_cli, box_length_cli, pairs_cli)

    if sum(1 for _ in mol_ref.topology.atoms()) != sum(1 for _ in mol_cli.topology.atoms()):
        raise RuntimeError("Atom count mismatch between reference and client-style paths.")

    grad_diff = np.asarray(grad_ref - grad_cli)
    energy_diff = float(energy_ref - energy_cli)
    virial_diff = float(virial_ref - virial_cli)

    pos_nm = np.array(pos_cli)
    client_virial_diag = np.diag((-np.asarray(grad_cli) * pos_nm).sum(axis=0) - float(virial_cli) * box_length_cli / 3.0).ravel()
    client_virial_si = np.array((client_virial_diag * kilojoule_per_mole / AVOGADRO_CONSTANT_NA).value_in_unit(joule))

    print(f"pairs_reference={pairs_ref.shape[0]}")
    print(f"pairs_client_style={pairs_cli.shape[0]}")
    print(f"polarization_solver={'fixed_'+str(args.pol_steps) if args.pol_steps is not None else 'converged'}")
    print(f"energy_reference_kjmol={float(energy_ref):.12f}")
    print(f"energy_client_style_kjmol={float(energy_cli):.12f}")
    print(f"energy_diff_kjmol={energy_diff:.12e}")
    print(f"grad_max_abs_diff_kjmol_per_nm={float(np.max(np.abs(grad_diff))):.12e}")
    print(f"grad_l2_diff_kjmol_per_nm={float(np.linalg.norm(grad_diff)):.12e}")
    print(f"box_grad_reference={float(virial_ref):.12f}")
    print(f"box_grad_client_style={float(virial_cli):.12f}")
    print(f"box_grad_diff={virial_diff:.12e}")
    print(f"client_ipi_virial_joule={client_virial_si}")


if __name__ == "__main__":
    main()
