# 3_MD

This directory contains the cleaned MD runtime for the current water-box workflow.

## Main Runtime Directory

- `CMD_H2O/`

## Bundled Dependency

- `vendor/DMFF/`

The MD scripts prefer the bundled `DMFF` source so the runtime does not depend on a separately installed local project checkout.

## Current Default Force Field

- `CMD_H2O/ff.backend_amoeba_total1000_classical_intra.xml`

This XML contains:

- the currently trained intermolecular model
- AMOEBA-style polarization damping in the bundled `DMFF`
- classical water intramolecular terms (`HarmonicBondForce` and `HarmonicAngleForce`)

## Main Commands

```bash
cd CMD_H2O
bash run.sh
```

Other useful commands:

```bash
bash run_nvt.sh
bash run_npt_soft.sh
python test_dmff.py
python compare_md_energy_paths.py
```
