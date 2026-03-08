#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBNAME="${1:-02molL_init.pdb}"
RUN_ID="${RUN_ID:-local_$(date +%Y%m%d_%H%M%S)}"
export RUN_ID
export SLURM_JOB_ID="${SLURM_JOB_ID:-${RUN_ID}}"

CONDA_ENV="${CONDA_ENV:-jaxmd}"
CONDA_SH="${CONDA_SH:-}"
SOCKET_PATH="/tmp/ipi_unix_dmff_${RUN_ID}"

find_conda_sh() {
  if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
    printf '%s\n' "${CONDA_SH}"
    return 0
  fi

  for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh"
  do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

cleanup() {
  local exit_code=$?
  if [[ -n "${CLIENT_PID:-}" ]] && kill -0 "${CLIENT_PID}" 2>/dev/null; then
    kill "${CLIENT_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${CLIENT_PID:-}" 2>/dev/null || true
  wait "${SERVER_PID:-}" 2>/dev/null || true
  rm -f "${SOCKET_PATH}"
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

if conda_sh_path="$(find_conda_sh)"; then
  # shellcheck disable=SC1090
  source "${conda_sh_path}"
  if conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV}"; then
    conda activate "${CONDA_ENV}"
  else
    echo "warning: conda env '${CONDA_ENV}' not found, using current shell environment" >&2
  fi
else
  echo "warning: conda.sh not found, using current shell environment" >&2
fi

command -v i-pi >/dev/null 2>&1 || {
  echo "error: i-pi not found in PATH" >&2
  exit 1
}
command -v python >/dev/null 2>&1 || {
  echo "error: python not found in PATH" >&2
  exit 1
}

if [[ ! -f "${SCRIPT_DIR}/${JOBNAME}" ]]; then
  echo "error: input PDB not found: ${SCRIPT_DIR}/${JOBNAME}" >&2
  exit 1
fi

rm -f /tmp/ipi_unix_dmff_*

echo "run_id=${RUN_ID}"
echo "jobname=${JOBNAME}"
echo "workdir=${SCRIPT_DIR}"
echo "start_time=$(date '+%F %T')"

cd "${SCRIPT_DIR}"

bash "${SCRIPT_DIR}/run_server.sh" "${JOBNAME}" &
SERVER_PID=$!

for _ in $(seq 1 60); do
  if [[ -S "${SOCKET_PATH}" ]]; then
    break
  fi
  sleep 1
done

if [[ ! -S "${SOCKET_PATH}" ]]; then
  echo "error: i-PI socket not ready: ${SOCKET_PATH}" >&2
  exit 1
fi

bash "${SCRIPT_DIR}/run_client_dmff.sh" "${JOBNAME}" &
CLIENT_PID=$!

wait "${CLIENT_PID}"
wait "${SERVER_PID}"

echo "finish_time=$(date '+%F %T')"
