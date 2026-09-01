#!/usr/bin/env bash
set -euo pipefail

SCRIPTPATH=$(dirname "$(readlink -f "$0")")
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"

CKPT_PATH="${CKPT_PATH:-weights/jamma.ckpt}"
IMC_ROOT="${IMC_ROOT:-data/imc}"
OUT_DIR="${OUT_DIR:-outputs/paper/imc}"
DUMP_DIR="${DUMP_DIR:-dump/det_jamma}"
DEVICE="${DEVICE:-cuda}"
BAG_SIZE="${BAG_SIZE:-5}"
DATA_CFG="${DATA_CFG:-configs/data/imc.py}"
MAIN_CFG="${MAIN_CFG:-configs/jamma/outdoor/test.py}"

SCENES=("$@")
if [ "${#SCENES[@]}" -eq 0 ]; then
  SCENES=(reichstag sacre_coeur st_peters_square)
fi

mkdir -p "${OUT_DIR}"

for SCENE in "${SCENES[@]}"; do
  SCENE_ROOT="${IMC_ROOT}/${SCENE}/set_100"

  python -u ./eval_imc.py \
    --ckpt_path "${CKPT_PATH}" \
    --methods nn-jamma det-jamma \
    --data_cfg_path "${DATA_CFG}" \
    --main_cfg_path "${MAIN_CFG}" \
    --dump_dir "${DUMP_DIR}/imc_${SCENE}" \
    --subset_dir "${SCENE_ROOT}/sub_set" \
    --dataset_root "${SCENE_ROOT}" \
    --calib_dir "${SCENE_ROOT}/calibration" \
    --bag_size "${BAG_SIZE}" \
    --dataset_name imc \
    --scene_name "${SCENE}" \
    --device "${DEVICE}" \
    --save_json "${OUT_DIR}/${SCENE}_tracking.json" \
    --save_summary_json "${OUT_DIR}/${SCENE}_tracking_summary.json"

  python -u ./test_imc_sfm.py \
    --method det-jamma \
    --jamma_ckpt "${CKPT_PATH}" \
    --data_cfg_path "${DATA_CFG}" \
    --main_cfg_path "${MAIN_CFG}" \
    --subset_dir "${SCENE_ROOT}/sub_set" \
    --dataset_root "${SCENE_ROOT}" \
    --calib_dir "${SCENE_ROOT}/calibration" \
    --bag_size "${BAG_SIZE}" \
    --dataset_name imc \
    --scene_name "${SCENE}" \
    --device "${DEVICE}" \
    --out_json "${OUT_DIR}/${SCENE}_sfm.json"
done
