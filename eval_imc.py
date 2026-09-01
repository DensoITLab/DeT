#!/usr/bin/env python3

import argparse
from pathlib import Path

from eval_tracking_common import add_common_tracking_args, run_tracking_evaluation


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_tracking_args(parser)
    parser.set_defaults(
        data_cfg_path="configs/data/imc.py",
        dataset_name="imc",
        save_json=Path("outputs/fig6_imc_tracking.json"),
        save_summary_json=Path("outputs/fig6_imc_tracking_summary.json"),
    )
    parser.add_argument("--imc_root", type=Path, default=Path("data/imc"))
    parser.add_argument("--scene_name", type=str, default="st_peters_square")
    parser.add_argument("--set_name", type=str, default="set_100")
    parser.add_argument("--subset_dir", type=Path, default=None)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--calib_dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    scene_root = args.imc_root / args.scene_name / args.set_name
    subset_dir = args.subset_dir or scene_root / "sub_set"
    dataset_root = args.dataset_root or scene_root
    calib_dir = args.calib_dir or scene_root / "calibration"
    run_tracking_evaluation(args, dataset_root=dataset_root, subset_dir=subset_dir, calib_dir=calib_dir)


if __name__ == "__main__":
    main()
