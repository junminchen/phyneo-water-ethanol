#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_XML="${BASE_XML:-${SCRIPT_DIR}/ff.backend_amoeba_total1000_classical_intra.xml}"
JOBNAME="${JOBNAME:-02molL_init.pdb}"
MODE="${MODE:-npt}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/thole_scan}"
XML_DIR="${OUTPUT_DIR}/xml"
LOG_DIR="${OUTPUT_DIR}/logs"
TOTAL_STEPS="${TOTAL_STEPS:-40000}"
PROP_STRIDE="${PROP_STRIDE:-10}"
TRAJ_STRIDE="${TRAJ_STRIDE:-100}"
CHECKPOINT_STRIDE="${CHECKPOINT_STRIDE:-200}"

mkdir -p "${XML_DIR}" "${LOG_DIR}"

if [[ "$#" -lt 1 ]]; then
  echo "usage: bash run_thole_scan.sh SCALE [SCALE ...]" >&2
  echo "example: bash run_thole_scan.sh 0.90 1.00 1.10" >&2
  exit 1
fi

for SCALE in "$@"; do
  SCALE_TAG="$(printf '%s' "${SCALE}" | tr '.-' 'pm')"
  FF_XML="${XML_DIR}/ff.thole_scale_${SCALE_TAG}.xml"
  RUN_TAG="thole_${SCALE_TAG}_${TIMESTAMP}"
  OUTPUT_PREFIX="out_${JOBNAME%.*}_${RUN_TAG}"

  python "${SCRIPT_DIR}/scale_thole_xml.py" \
    --input "${BASE_XML}" \
    --output "${FF_XML}" \
    --scale "${SCALE}" \
    | tee "${LOG_DIR}/${RUN_TAG}.scale.log"

  export FF_XML
  export OUTPUT_PREFIX
  export RUN_ID="${RUN_TAG}"
  export TOTAL_STEPS
  export PROP_STRIDE
  export TRAJ_STRIDE
  export CHECKPOINT_STRIDE

  echo "run_id=${RUN_ID}"
  echo "ff_xml=${FF_XML}"
  echo "output_prefix=${OUTPUT_PREFIX}"
  echo "total_steps=${TOTAL_STEPS}"
  echo "prop_stride=${PROP_STRIDE}"
  echo "traj_stride=${TRAJ_STRIDE}"
  echo "checkpoint_stride=${CHECKPOINT_STRIDE}"

  if [[ "${MODE}" == "nvt" ]]; then
    bash "${SCRIPT_DIR}/run_nvt.sh" "${JOBNAME}"
  else
    bash "${SCRIPT_DIR}/run.sh" "${JOBNAME}"
  fi
done
