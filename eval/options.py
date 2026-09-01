import argparse
import math
from pathlib import Path


def add_tracking_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data_cfg_path", type=str, default="configs/data/megadepth_test_1500.py")
    parser.add_argument("--main_cfg_path", type=str, default="configs/jamma/outdoor/test.py")
    parser.add_argument("--ckpt_path", type=str, default="weights/jamma.ckpt")
    parser.add_argument("--dump_dir", type=str, default="dump/eval_tracking")
    parser.add_argument("--profiler_name", type=str, default="inference")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--methods", type=str, nargs="+", default=["nn-jamma", "det-jamma"])
    parser.add_argument("--bag_size", type=int, default=5)
    parser.add_argument("--topk", type=int, default=20000)
    parser.add_argument("--epi_thr", type=float, default=1e-3)
    parser.add_argument("--nn_link_radius", type=float, default=5.0 * math.sqrt(2.0))
    parser.add_argument("--det_search_radius", type=float, default=832.0 * math.sqrt(2.0))
    parser.add_argument("--det_fine_thr", type=float, default=0.0)
    parser.add_argument("--custom_fine_flex_thr", type=float, default=0.1)
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--df", type=int, default=None)
    parser.add_argument("--no_padding", action="store_true")
    parser.add_argument("--thr", type=float, default=None)
    parser.add_argument("--flip_w2c", action="store_true")
    parser.add_argument("--save_json", type=Path, default=None)
    parser.add_argument("--save_summary_json", type=Path, default=None)
    parser.add_argument("--save_errors", action="store_true")
    parser.add_argument("--dataset_name", type=str, default=None)
    return parser
