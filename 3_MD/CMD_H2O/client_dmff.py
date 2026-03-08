#!/usr/bin/env python3
import os
import sys
from pathlib import Path

MD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_DMFF_ROOT = MD_ROOT / "vendor" / "DMFF"
LOCAL_DMFF_ROOT = REPO_ROOT / "DMFF"
for dmff_root in (BUNDLED_DMFF_ROOT, LOCAL_DMFF_ROOT):
    if dmff_root.exists() and str(dmff_root) not in sys.path:
        sys.path.insert(0, str(dmff_root))

import driver
import numpy as np
from openmm import app
from openmm.unit import AVOGADRO_CONSTANT_NA, joule, kilojoule_per_mole, meter, nanometer

import jax.numpy as jnp
from jax import jit, value_and_grad

from dmff.api import Hamiltonian
from dmff.common import nblist


class DMFFDriver(driver.BaseDriver):
    def __init__(self, addr, port, socktype, pdb, ff_xml, r_xml, *_unused):
        run_id = os.environ.get("RUN_ID", os.environ.get("SLURM_JOB_ID", "local"))
        addr = f"{addr}_{run_id}"
        super().__init__(port, addr, socktype)

        pdb_path = Path(pdb).resolve()
        ff_xml = str(Path(ff_xml).resolve())
        r_xml = str(Path(r_xml).resolve())

        app.Topology.loadBondDefinitions(r_xml)
        mol = app.PDBFile(str(pdb_path))
        pos = jnp.array(mol.positions._value)
        box = jnp.array(mol.topology.getPeriodicBoxVectors()._value)
        L = box[0][0]

        rc = 0.6
        H = Hamiltonian(ff_xml)
        potential_kwargs = {
            "nonbondedCutoff": rc * nanometer,
            "nonbondedMethod": app.PME,
            "ethresh": 1e-4,
        }
        pol_steps_env = os.environ.get("POL_STEPS", "20").strip()
        if pol_steps_env:
            potential_kwargs["step_pol"] = int(pol_steps_env)
        pots = H.createPotential(mol.topology, **potential_kwargs)
        efunc = pots.getPotentialFunc()
        params = H.getParameters()
        print(
            f"polarization_solver={'fixed_'+pol_steps_env if pol_steps_env else 'converged'}",
            flush=True,
        )

        self.nbl = nblist.NeighborListFreud(box, rc, pots.meta["cov_map"], padding=False)
        self.nbl.allocate(pos, box)
        pairs = self.nbl.pairs
        print(
            f"neighbor_list total={pairs.shape[0]} real={pairs.shape[0]} padded=0",
            flush=True,
        )

        def dmff_calculator(pos_nm, box_length_nm, pair_index):
            box_nm = jnp.array(
                [
                    [box_length_nm, 0.0, 0.0],
                    [0.0, box_length_nm, 0.0],
                    [0.0, 0.0, box_length_nm],
                ]
            )
            return efunc(pos_nm, box_nm, pair_index, params)

        self.calc_dmff = jit(value_and_grad(dmff_calculator, argnums=(0, 1)))

        energy, (grad, virial) = self.calc_dmff(pos, L, pairs)
        print(energy, grad, virial)

    def grad(self, crd, cell):
        pos = np.array(crd * 1e9)
        box = np.array(cell * 1e9)
        L = box[0][0]

        self.nbl.update(pos, box)
        pairs = self.nbl.pairs

        energy, (grad, virial) = self.calc_dmff(pos, L, pairs)
        virial = np.diag((-grad * pos).sum(axis=0) - virial * L / 3.0).ravel()

        energy = np.array((energy * kilojoule_per_mole / AVOGADRO_CONSTANT_NA).value_in_unit(joule))
        grad = np.array((grad * kilojoule_per_mole / nanometer / AVOGADRO_CONSTANT_NA).value_in_unit(joule / meter))
        virial = np.array((virial * kilojoule_per_mole / AVOGADRO_CONSTANT_NA).value_in_unit(joule))
        return energy, grad, virial


if __name__ == "__main__":
    addr = sys.argv[1]
    port = int(sys.argv[2])
    socktype = sys.argv[3]
    fn_pdb = sys.argv[4]
    ff_xml = sys.argv[5]
    r_xml = sys.argv[6]
    fn_psr = sys.argv[7] if len(sys.argv) > 7 else ""
    fn_psr_ABn = sys.argv[8] if len(sys.argv) > 8 else ""

    driver_dmff = DMFFDriver(addr, port, socktype, fn_pdb, ff_xml, r_xml, fn_psr, fn_psr_ABn)
    while True:
        driver_dmff.parse()
