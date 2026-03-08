#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
CHECK_DIR="${SCRIPT_DIR}/checks/latest"
CHECKPOINT="${CHECKPOINT:-${SCRIPT_DIR}/checkpoints/latest.pickle}"
DATA_FILE="${SCRIPT_DIR}/data/data_sr_amoeba.pickle"

mkdir -p "${LOG_DIR}" "${CHECK_DIR}"

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

cd "${PROJECT_ROOT}"
python check_trained_dimer_scans.py \
  --data "${DATA_FILE}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${CHECK_DIR}" \
  "$@" | tee "${LOG_DIR}/check.log"
