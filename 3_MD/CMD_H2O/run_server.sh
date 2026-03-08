#!/bin/bash
export OMP_NUM_THREADS=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
jobname=$1
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-local}}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-${SCRIPT_DIR}/input.xml}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-out_${jobname%.*}}"
TOTAL_STEPS="${TOTAL_STEPS:-}"
PROP_STRIDE="${PROP_STRIDE:-}"
TRAJ_STRIDE="${TRAJ_STRIDE:-}"
CHECKPOINT_STRIDE="${CHECKPOINT_STRIDE:-}"
TIMESTEP_FS="${TIMESTEP_FS:-}"

sed_args=(
  -E
  -e "s#<address>[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*</address>#<address> \\1_${RUN_ID} </address>#"
  -e "s/simulation/${OUTPUT_PREFIX}/g"
  -e "s/init_ipi.pdb/${jobname%.*}_ipi.pdb/g"
)

if [[ -n "${TOTAL_STEPS}" ]]; then
  sed_args+=(-e "s#<total_steps>[[:space:]]*[0-9]+[[:space:]]*</total_steps>#<total_steps>${TOTAL_STEPS}</total_steps>#")
fi
if [[ -n "${PROP_STRIDE}" ]]; then
  sed_args+=(-e "s#<properties filename='out' stride='[0-9]+'#<properties filename='out' stride='${PROP_STRIDE}'#")
fi
if [[ -n "${TRAJ_STRIDE}" ]]; then
  sed_args+=(-e "s#<trajectory filename='pos' stride='[0-9]+'#<trajectory filename='pos' stride='${TRAJ_STRIDE}'#")
  sed_args+=(-e "s#<trajectory filename='for' stride='[0-9]+'#<trajectory filename='for' stride='${TRAJ_STRIDE}'#")
fi
if [[ -n "${CHECKPOINT_STRIDE}" ]]; then
  sed_args+=(-e "s#<checkpoint stride='[0-9]+'#<checkpoint stride='${CHECKPOINT_STRIDE}'#")
fi
if [[ -n "${TIMESTEP_FS}" ]]; then
  sed_args+=(-e "s#<timestep units='femtosecond'>[[:space:]]*[0-9.]+[[:space:]]*</timestep>#<timestep units='femtosecond'> ${TIMESTEP_FS} </timestep>#")
fi

sed "${sed_args[@]}" "${INPUT_TEMPLATE}" > "${SCRIPT_DIR}/${RUN_ID}.tmp.xml"
# i-pi simulation_lj.restart >& logfile &
i-pi "${SCRIPT_DIR}/${RUN_ID}.tmp.xml" >& "${SCRIPT_DIR}/${RUN_ID}.log" &
#i-pi simulation.restart >& logfile &

# i-pi input_nvt.xml >& logfile &

wait
