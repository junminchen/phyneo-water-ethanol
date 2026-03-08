#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODE="${MODE:-nvt}"
export TOTAL_STEPS="${TOTAL_STEPS:-20000}"
export TIMESTEP_FS="${TIMESTEP_FS:-0.1}"
export PROP_STRIDE="${PROP_STRIDE:-20}"
export TRAJ_STRIDE="${TRAJ_STRIDE:-200}"
export CHECKPOINT_STRIDE="${CHECKPOINT_STRIDE:-500}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/thole_stability_scan}"

echo "mode=${MODE}"
echo "total_steps=${TOTAL_STEPS}"
echo "timestep_fs=${TIMESTEP_FS}"
echo "prop_stride=${PROP_STRIDE}"
echo "traj_stride=${TRAJ_STRIDE}"
echo "checkpoint_stride=${CHECKPOINT_STRIDE}"
echo "output_dir=${OUTPUT_DIR}"

bash "${SCRIPT_DIR}/run_thole_scan.sh" "$@"
