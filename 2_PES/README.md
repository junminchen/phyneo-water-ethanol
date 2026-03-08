# 2_PES

This directory contains the maintained potential-energy-surface workflow used in the cleaned repository.

Only the current AMOEBA-damping training branch is kept here.

## Included Workflow

- `training_backend_amoeba_damping/`
  - rebuild long-range subtraction from `data.pickle`
  - train the short-range shared-parameter model
  - run dimer-check plots and XML export

## Why old scripts are omitted

The original workspace contained many historical notebooks, intermediate datasets, and old fitting branches.

For GitHub, the retained workflow is only the one that is internally consistent with the current MD model:

- AMOEBA-style polarization damping in bundled `DMFF`
- regenerated long-range subtraction
- retrained short-range parameters

## Main Outputs

- `training_backend_amoeba_damping/data/data_sr_amoeba.pickle`
- `training_backend_amoeba_damping/data/data_lr_amoeba.pickle`
- `training_backend_amoeba_damping/checkpoints/latest.pickle`
- `training_backend_amoeba_damping/checkpoints/latest.xml`
- `training_backend_amoeba_damping/checks/latest/trained_params.xml`
- `training_backend_amoeba_damping/checks/latest/rmsd_summary.json`
