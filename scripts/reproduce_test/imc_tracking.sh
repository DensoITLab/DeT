#!/usr/bin/env bash
set -euo pipefail

python -u ./test_imc_tracking.py \
  --ckpt_path /path/to/det.ckpt \
  --methods jamma \
  --subset_dir data/imc/st_peters_square/set_100/sub_set \
  --dataset_root data/imc/st_peters_square/set_100 \
  --calib_dir data/imc/st_peters_square/set_100/calibration \
  --depth_dir data/imc/st_peters_square/set_100/depth_maps \
  --save_summary_json outputs/fig6_imc_tracking_summary.json
