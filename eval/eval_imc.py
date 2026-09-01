#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.options import add_tracking_args


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_tracking_args(parser)
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

    from eval.tracking import run_tracking_evaluation

    run_tracking_evaluation(args, dataset_root=dataset_root, subset_dir=subset_dir, calib_dir=calib_dir)


if __name__ == "__main__":
    main()
