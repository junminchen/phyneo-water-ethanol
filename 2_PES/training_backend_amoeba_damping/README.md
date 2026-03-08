# Dimer Training With AMOEBA-Style Polarization Damping

This folder is the clean entry point for the maintained dimer fitting workflow.

## What It Does

1. Rebuild the long-range subtraction with the current local `DMFF`
2. Retrain the faithful short-range shared-parameter model
3. Check the resulting dimer scans and export a trained XML

## Files

- `prepare_data.py`
  - rebuild `data_sr_amoeba.pickle` and `data_lr_amoeba.pickle`
- `run_prepare.sh`
  - shell wrapper for the long-range subtraction step
- `run_train.sh`
  - shell wrapper for backend short-range training
- `run_check.sh`
  - shell wrapper for dimer-check plotting and XML export
- `data/`
  - processed datasets used by the current model
- `checkpoints/`
  - current tracked checkpoint and rendered XML
- `checks/latest/`
  - current tracked check outputs

## Default Trainable Parameters

- `A_ex`
- `A_es`
- `A_pol`
- `A_disp`
- `A_dhf`
- shared `B`

`thole` is not part of this short-range training objective.

## Commands

```bash
bash run_prepare.sh
bash run_train.sh
bash run_check.sh
```

Override epochs if needed:

```bash
EPOCHS=200 bash run_train.sh
```
