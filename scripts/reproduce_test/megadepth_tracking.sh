#!/usr/bin/env bash
set -euo pipefail

SCRIPTPATH=$(dirname "$(readlink -f "$0")")
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"

CKPT_PATH="${CKPT_PATH:-/path/to/det.ckpt}"
MEGADEPTH_ROOT="${MEGADEPTH_ROOT:-data/megadepth}"
MEGADEPTH_SFM_ROOT="${MEGADEPTH_SFM_ROOT:-${MEGADEPTH_ROOT}/Undistorted_SfM}"
SCENE_RAW="${SCENE:-0022}"
OUT_JSON="${OUT_JSON:-outputs/fig6_megadepth_tracking.json}"
OUT_SUMMARY_JSON="${OUT_SUMMARY_JSON:-outputs/fig6_megadepth_tracking_summary.json}"
DEVICE="${DEVICE:-cuda}"
METHODS="${METHODS:-det-jamma}"
DATA_CFG="${DATA_CFG:-configs/data/megadepth_test_1500.py}"
MAIN_CFG="${MAIN_CFG:-configs/jamma/outdoor/test.py}"

case "${SCENE_RAW}" in
  15) SCENE="0015" ;;
  22) SCENE="0022" ;;
  *) SCENE="${SCENE_RAW}" ;;
esac

SCENE_ROOT="${MEGADEPTH_SFM_ROOT}/${SCENE}"
DUMP_DIR="${DUMP_DIR:-dump/det_jamma/megadepth_${SCENE}}"

python -u ./test_megadepth_tracking.py \
  --ckpt_path "${CKPT_PATH}" \
  --methods ${METHODS} \
  --data_cfg_path "${DATA_CFG}" \
  --main_cfg_path "${MAIN_CFG}" \
  --dump_dir "${DUMP_DIR}" \
  --subset_dir "${SCENE_ROOT}/5bag" \
  --dataset_root "${MEGADEPTH_ROOT}" \
  --calib_dir "${SCENE_ROOT}/calibration" \
  --depth_dir "${MEGADEPTH_SFM_ROOT}/depth_undistorted/${SCENE}" \
  --bag_size 5 \
  --dataset_name megadepth \
  --scene_name "${SCENE}" \
  --device "${DEVICE}" \
  --save_json "${OUT_JSON}" \
  --save_summary_json "${OUT_SUMMARY_JSON}"
