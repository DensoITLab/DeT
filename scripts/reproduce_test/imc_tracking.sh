#!/usr/bin/env bash
set -euo pipefail

SCRIPTPATH=$(dirname "$(readlink -f "$0")")
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"

CKPT_PATH="${CKPT_PATH:-weights/jamma.ckpt}"
IMC_ROOT="${IMC_ROOT:-data/imc}"
SCENE="${SCENE:-st_peters_square}"
SET_NAME="${SET_NAME:-set_100}"
OUT_JSON="${OUT_JSON:-outputs/fig6_imc_tracking.json}"
OUT_SUMMARY_JSON="${OUT_SUMMARY_JSON:-outputs/fig6_imc_tracking_summary.json}"
DUMP_DIR="${DUMP_DIR:-dump/det_jamma/imc_${SCENE}}"
DEVICE="${DEVICE:-cuda}"
METHODS="${METHODS:-nn-jamma det-jamma}"
DATA_CFG="${DATA_CFG:-configs/data/imc.py}"
MAIN_CFG="${MAIN_CFG:-configs/jamma/outdoor/test.py}"
SCENE_ROOT="${IMC_ROOT}/${SCENE}/${SET_NAME}"

python -u ./eval_imc.py \
  --ckpt_path "${CKPT_PATH}" \
  --methods ${METHODS} \
  --data_cfg_path "${DATA_CFG}" \
  --main_cfg_path "${MAIN_CFG}" \
  --dump_dir "${DUMP_DIR}" \
  --subset_dir "${SCENE_ROOT}/sub_set" \
  --dataset_root "${SCENE_ROOT}" \
  --calib_dir "${SCENE_ROOT}/calibration" \
  --bag_size 5 \
  --dataset_name imc \
  --scene_name "${SCENE}" \
  --device "${DEVICE}" \
  --save_json "${OUT_JSON}" \
  --save_summary_json "${OUT_SUMMARY_JSON}"
