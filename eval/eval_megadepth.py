#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.options import add_tracking_args


def normalize_scene(scene_name: str) -> str:
    if scene_name == "15":
        return "0015"
    if scene_name == "22":
        return "0022"
    return scene_name


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_tracking_args(parser)
    parser.set_defaults(
        data_cfg_path="configs/data/megadepth_test_1500.py",
        dataset_name="megadepth",
        save_json=Path("outputs/fig6_megadepth_tracking.json"),
        save_summary_json=Path("outputs/fig6_megadepth_tracking_summary.json"),
    )
    parser.add_argument("--megadepth_root", type=Path, default=Path("data/megadepth"))
    parser.add_argument("--megadepth_sfm_root", type=Path, default=None)
    parser.add_argument("--scene_name", type=str, default="0022")
    parser.add_argument("--subset_dir", type=Path, default=None)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--calib_dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.scene_name = normalize_scene(args.scene_name)
    sfm_root = args.megadepth_sfm_root or args.megadepth_root / "Undistorted_SfM"
    scene_root = sfm_root / args.scene_name
    subset_dir = args.subset_dir or scene_root / "5bag"
    dataset_root = args.dataset_root or args.megadepth_root
    calib_dir = args.calib_dir or scene_root / "calibration"

    from eval.tracking import run_tracking_evaluation

    run_tracking_evaluation(args, dataset_root=dataset_root, subset_dir=subset_dir, calib_dir=calib_dir)


if __name__ == "__main__":
    main()
