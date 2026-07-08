#!/usr/bin/env bash
set -euo pipefail

python -u ./test_megadepth_tracking.py \
  --ckpt_path /path/to/det.ckpt \
  --methods jamma \
  --subset_dir data/megadepth/Undistorted_SfM/0022/5bag \
  --dataset_root data/megadepth \
  --calib_dir data/megadepth/Undistorted_SfM/0022/calibration \
  --depth_dir data/megadepth/Undistorted_SfM/depth_undistorted/0022 \
  --save_summary_json outputs/fig6_megadepth_tracking_summary.json
