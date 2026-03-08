#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
SAVE_DIR="${SCRIPT_DIR}/checkpoints"
DATA_FILE="${SCRIPT_DIR}/data/data_sr_amoeba.pickle"

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

if [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
  source /opt/anaconda3/etc/profile.d/conda.sh
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/opt/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/opt/anaconda3/etc/profile.d/conda.sh"
fi

CONDA_ENV="${CONDA_ENV:-jaxmd}"
if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV}"
fi

EPOCHS="${EPOCHS:-1000}"
LR="${LR:-1e-2}"
SAVE_EVERY="${SAVE_EVERY:-100}"
REPORT_EVERY="${REPORT_EVERY:-1}"
SEED="${SEED:-20260308}"

cd "${PROJECT_ROOT}"
python train_dimer_backend.py \
  --data "${DATA_FILE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --optimizer simple_adam \
  --save-dir "${SAVE_DIR}" \
  --save-every "${SAVE_EVERY}" \
  --report-every "${REPORT_EVERY}" \
  --write-xml \
  "$@" | tee "${LOG_DIR}/train.log"
