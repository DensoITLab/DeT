#!/usr/bin/env bash
set -euo pipefail

SCRIPTPATH=$(dirname "$(readlink -f "$0")")
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"

CKPT_PATH="${CKPT_PATH:-/path/to/det.ckpt}"
MEGADEPTH_ROOT="${MEGADEPTH_ROOT:-data/megadepth}"
MEGADEPTH_SFM_ROOT="${MEGADEPTH_SFM_ROOT:-${MEGADEPTH_ROOT}/Undistorted_SfM}"
OUT_DIR="${OUT_DIR:-outputs/paper/megadepth}"
DUMP_DIR="${DUMP_DIR:-dump/det_jamma}"
DEVICE="${DEVICE:-cuda}"
BAG_SIZE="${BAG_SIZE:-5}"
DATA_CFG="${DATA_CFG:-configs/data/megadepth_test_1500.py}"
MAIN_CFG="${MAIN_CFG:-configs/jamma/outdoor/test.py}"

SCENES=("$@")
if [ "${#SCENES[@]}" -eq 0 ]; then
  SCENES=(0015 0022)
fi

mkdir -p "${OUT_DIR}"

normalize_scene() {
  case "$1" in
    15) printf "0015" ;;
    22) printf "0022" ;;
    *) printf "%s" "$1" ;;
  esac
}

for SCENE_RAW in "${SCENES[@]}"; do
  SCENE=$(normalize_scene "${SCENE_RAW}")
  SCENE_ROOT="${MEGADEPTH_SFM_ROOT}/${SCENE}"

  python -u ./test_megadepth_tracking.py \
    --ckpt_path "${CKPT_PATH}" \
    --methods det-jamma \
    --data_cfg_path "${DATA_CFG}" \
    --main_cfg_path "${MAIN_CFG}" \
    --dump_dir "${DUMP_DIR}/megadepth_${SCENE}" \
    --subset_dir "${SCENE_ROOT}/5bag" \
    --dataset_root "${MEGADEPTH_ROOT}" \
    --calib_dir "${SCENE_ROOT}/calibration" \
    --depth_dir "${MEGADEPTH_SFM_ROOT}/depth_undistorted/${SCENE}" \
    --bag_size "${BAG_SIZE}" \
    --dataset_name megadepth \
    --scene_name "${SCENE}" \
    --device "${DEVICE}" \
    --save_json "${OUT_DIR}/${SCENE}_tracking.json" \
    --save_summary_json "${OUT_DIR}/${SCENE}_tracking_summary.json"

  python -u ./test_imc_sfm.py \
    --method det-jamma \
    --jamma_ckpt "${CKPT_PATH}" \
    --data_cfg_path "${DATA_CFG}" \
    --main_cfg_path "${MAIN_CFG}" \
    --subset_dir "${SCENE_ROOT}/5bag" \
    --dataset_root "${MEGADEPTH_ROOT}" \
    --calib_dir "${SCENE_ROOT}/calibration" \
    --bag_size "${BAG_SIZE}" \
    --dataset_name megadepth \
    --scene_name "${SCENE}" \
    --device "${DEVICE}" \
    --out_json "${OUT_DIR}/${SCENE}_sfm.json"
done
