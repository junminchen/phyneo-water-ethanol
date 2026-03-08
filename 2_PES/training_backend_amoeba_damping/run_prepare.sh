#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
DATA_DIR="${SCRIPT_DIR}/data"

mkdir -p "${LOG_DIR}" "${DATA_DIR}"

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
python "${SCRIPT_DIR}/prepare_data.py" \
  --output-sr "${DATA_DIR}/data_sr_amoeba.pickle" \
  --output-lr "${DATA_DIR}/data_lr_amoeba.pickle" \
  "$@" | tee "${LOG_DIR}/prepare.log"
