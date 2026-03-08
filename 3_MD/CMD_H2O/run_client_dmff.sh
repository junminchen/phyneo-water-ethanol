#!/bin/bash
export OPENMM_CPU_THREADS=1
export OMP_NUM_THREADS=1
export POL_STEPS="${POL_STEPS:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUNDLED_DMFF_ROOT="${PROJECT_ROOT}/3_MD/vendor/DMFF"
LOCAL_DMFF_ROOT="${PROJECT_ROOT}/DMFF"

addr=unix_dmff
port=1234
socktype=unix
jobname=$1
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-local}}"

# Latest MD-test force field built with AMOEBA-style polarization damping
# and classical water intra terms injected into the trained XML.
ff_xml="${FF_XML:-${SCRIPT_DIR}/ff.backend_amoeba_total1000_classical_intra.xml}"
r_xml="${SCRIPT_DIR}/residues.xml"
if [[ -d "${BUNDLED_DMFF_ROOT}" ]]; then
  export PYTHONPATH="${BUNDLED_DMFF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
elif [[ -d "${LOCAL_DMFF_ROOT}" ]]; then
  export PYTHONPATH="${LOCAL_DMFF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

python "${SCRIPT_DIR}/client_dmff.py" \
  "$addr" \
  "$port" \
  "$socktype" \
  "${SCRIPT_DIR}/${jobname}" \
  "$ff_xml" \
  "$r_xml" \
  > "${SCRIPT_DIR}/${RUN_ID}.log_dmff"
