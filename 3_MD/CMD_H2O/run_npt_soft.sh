#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export INPUT_TEMPLATE="${SCRIPT_DIR}/input_npt_soft.xml"

bash "${SCRIPT_DIR}/run.sh" "$@"
