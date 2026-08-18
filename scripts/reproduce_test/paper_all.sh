#!/usr/bin/env bash
set -euo pipefail

SCRIPTPATH=$(dirname "$(readlink -f "$0")")
PROJECT_DIR="${SCRIPTPATH}/../../"

cd "${PROJECT_DIR}"

bash scripts/reproduce_test/paper_imc.sh
bash scripts/reproduce_test/paper_megadepth_0015_0022.sh
