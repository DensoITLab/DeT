#!/usr/bin/env bash
set -euo pipefail

SCRIPTPATH=$(dirname "$(readlink -f "$0")")
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"

CKPT_PATH="${CKPT_PATH:-/path/to/det.ckpt}"
IMC_ROOT="${IMC_ROOT:-data/imc}"
SCENE="${SCENE:-st_peters_square}"
SET_NAME="${SET_NAME:-set_100}"
OUT_JSON="${OUT_JSON:-outputs/imc_sfm_det_jamma.json}"
DEVICE="${DEVICE:-cuda}"
METHOD="${METHOD:-det-jamma}"
DATA_CFG="${DATA_CFG:-configs/data/imc.py}"
MAIN_CFG="${MAIN_CFG:-configs/jamma/outdoor/test.py}"
SCENE_ROOT="${IMC_ROOT}/${SCENE}/${SET_NAME}"

python -u ./test_imc_sfm.py \
    --method "${METHOD}" \
    --jamma_ckpt "${CKPT_PATH}" \
    --subset_dir "${SCENE_ROOT}/sub_set" \
    --dataset_root "${SCENE_ROOT}" \
    --calib_dir "${SCENE_ROOT}/calibration" \
    --data_cfg_path "${DATA_CFG}" \
    --main_cfg_path "${MAIN_CFG}" \
    --bag_size 5 \
    --dataset_name imc \
    --scene_name "${SCENE}" \
    --device "${DEVICE}" \
    --out_json "${OUT_JSON}"
