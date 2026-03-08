# project_h2o_etoh

Clean GitHub-ready subset of the original `project_h2o_etoh` workspace.

This repository keeps only the parts needed to:

1. rebuild the long-range subtraction,
2. retrain the short-range dimer model,
3. inspect dimer scans, and
4. run i-PI + DMFF molecular dynamics with the current force field.

Historical notebooks, old trial branches, temporary logs, and large exploratory folders are intentionally left out.

## Current Model

- Base force field: `ff.xml`
- Current training lineage: `2_PES/training_backend_amoeba_damping`
- Current MD default XML:
  - `3_MD/CMD_H2O/ff.backend_amoeba_total1000_classical_intra.xml`
- Polarization model:
  - AMOEBA-style Thole damping in bundled `DMFF`
  - implemented in `3_MD/vendor/DMFF/dmff/admp/pme.py`

## Repository Layout

- `train_dimer_backend.py`
  - faithful backend short-range training script
- `check_trained_dimer_scans.py`
  - render trained XML and make dimer-check plots
- `ff.xml`
  - base force field used for retraining
- `data.pickle`
  - raw dimer reference data before long-range subtraction
- `dimer/`
  - dimer PDB templates
- `monomer/`
  - monomer PDB templates
- `2_PES/training_backend_amoeba_damping/`
  - clean workflow for long-range subtraction, retraining, and dimer checking
- `3_MD/CMD_H2O/`
  - MD inputs, run scripts, and current trained XMLs
- `3_MD/vendor/DMFF/`
  - bundled patched DMFF runtime used by MD and available as a training fallback

## Training Workflow

Go to:

```bash
cd 2_PES/training_backend_amoeba_damping
```

Rebuild the long-range subtraction:

```bash
bash run_prepare.sh
```

Train:

```bash
bash run_train.sh
```

Check the trained model:

```bash
bash run_check.sh
```

Key tracked outputs:

- `data/data_sr_amoeba.pickle`
- `data/data_lr_amoeba.pickle`
- `checkpoints/latest.pickle`
- `checkpoints/latest.xml`
- `checks/latest/trained_params.xml`
- `checks/latest/rmsd_summary.json`

## MD Workflow

Go to:

```bash
cd 3_MD/CMD_H2O
```

Default NPT:

```bash
bash run.sh
```

NVT:

```bash
bash run_nvt.sh
```

Softer NPT:

```bash
bash run_npt_soft.sh
```

Single-point sanity check:

```bash
python test_dmff.py
```

Energy-path consistency check:

```bash
python compare_md_energy_paths.py
```

## Dependencies

Recommended environment:

- Python 3.10+
- JAX
- OpenMM
- i-PI for MD runs

Important:

- This repository prefers the bundled local `DMFF` source over any site-installed `dmff`.
- `3_MD/CMD_H2O` uses `3_MD/vendor/DMFF` by default.
- `train_dimer_backend.py` falls back to the same bundled `DMFF` if top-level `DMFF/` is absent.

## Notes

- The short-range training objective does **not** train `thole`.
- `thole` belongs to the long-range polarization model, so changing the damping form requires rebuilding the long-range subtraction first.
- The MD XML includes classical water intramolecular terms (`HarmonicBondForce` and `HarmonicAngleForce`) injected into the trained intermolecular XML.
