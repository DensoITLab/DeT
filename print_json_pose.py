#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import argparse


# ============================================================
#  PAIRS (pairwise epipolar)
# ============================================================
def dump_pairs_epi(lines, method_name, msum):
    ps = msum.get("pairs_summary")
    if ps is None:
        return

    lines.append("")
    lines.append("# PAIRS_EPI: pair-wise epipolar metrics")
    lines.append("table,method,metric,value")

    for k, v in ps.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                lines.append(f"PAIRS_EPI,{method_name},{k}.{kk},{vv}")
        else:
            lines.append(f"PAIRS_EPI,{method_name},{k},{v}")


# ============================================================
#  EPI 0→N
# ============================================================
def dump_epi_0N(lines, method_name, msum):
    s = msum.get("0N_summary")
    if s is None:
        return

    lines.append("")
    lines.append("# EPI_0N: 0→N epipolar metrics")
    lines.append("table,method,metric,value")

    # precision（1e-4, 5e-4）
    for key in [
        "global_precision_0N@1e-4_by_counts",
        "global_precision_0N@5e-4_by_counts",
        "auc_0N@5",
        "auc_0N@10",
        "auc_0N@20",
    ]:
        if key in s:
            lines.append(f"EPI_0N,{method_name},{key},{s[key]}")

    # 分子・分母
    gc = s.get("global_counts_0N", {})
    for k, v in gc.items():
        lines.append(f"EPI_0N,{method_name},global_counts_0N.{k},{v}")

    # survival
    ts = s.get("global_track_survival_0_to_N", {})
    for k, v in ts.items():
        lines.append(f"EPI_0N,{method_name},track_survival.{k},{v}")

    # bags
    for key in ["bags_with_valid_pose_0N", "bags_total"]:
        if key in s:
            lines.append(f"EPI_0N,{method_name},{key},{s[key]}")


# ============================================================
#  EPI 0→k
# ============================================================
def dump_epi_0K(lines, method_name, msum):
    s = msum.get("0k_summary")
    if s is None:
        return

    lines.append("")
    lines.append("# EPI_0K: 0→k epipolar metrics per k")
    lines.append("table,method,mode,k,precision@1e-4,precision@5e-4,correct_1e-4,correct_5e-4,total,median_err")

    for k_str in sorted(s.keys(), key=lambda x: int(x)):
        info = s[k_str]

        prec1 = info.get("precision@1e-4_0_to_k", float("nan"))
        prec5 = info.get("precision@5e-4_0_to_k", float("nan"))

        gc = info.get("global_counts_0_to_k", {})

        correct1 = int(gc.get("correct", 0))
        correct5 = int(gc.get("correct_5e-4", 0))
        total = int(gc.get("total", 0))

        med_err = float(info.get("median_err_0_to_k", float("nan")))
        k = int(k_str)

        lines.append(
            f"EPI_0K,{method_name},0->k,{k},"
            f"{prec1},{prec5},{correct1},{correct5},{total},{med_err}"
        )


# ============================================================
#  EQUALIZED 0→N
# ============================================================
def dump_epi_equalized_0N(lines, method_name, msum):
    s = msum.get("0N_equalized_min_tracks")
    if s is None:
        return

    lines.append("")
    lines.append("# EPI_0N_EQUALIZED: equalized")
    lines.append("table,method,metric,value")

    for key in [
        "global_precision_0N_equalized_min_tracks_conf_sum",
        "global_precision_0N_equalized_min_tracks_conf_prod",
    ]:
        if key in s:
            lines.append(f"EPI_0N_EQUALIZED,{method_name},{key},{s[key]}")

    counts_sum = s.get("global_counts_0N_equalized_min_tracks_conf_sum", {})
    for k, v in counts_sum.items():
        lines.append(f"EPI_0N_EQUALIZED,{method_name},counts_sum.{k},{v}")

    counts_prod = s.get("global_counts_0N_equalized_min_tracks_conf_prod", {})
    for k, v in counts_prod.items():
        lines.append(f"EPI_0N_EQUALIZED,{method_name},counts_prod.{k},{v}")


# ============================================================
#  FLOPS / TIME
# ============================================================
def dump_flops_time(lines, method_name, msum):
    ft = msum.get("flops_time_summary")
    if ft is None:
        return

    lines.append("")
    lines.append("# FLOPS_TIME")
    lines.append("table,method,metric,value")

    for k, v in ft.items():
        lines.append(f"FLOPS_TIME,{method_name},{k},{v}")


# ============================================================
#  MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=Path("/home/ach17765lb/data/phototourism/results_bag_eval_summary.json"),
    )
    parser.add_argument(
        "--out_txt",
        type=Path,
        default=Path("/home/ach17765lb/data/phototourism/results_bag_eval_epi_summary.txt"),
    )
    args = parser.parse_args()

    summary = json.loads(args.summary_json.read_text())

    lines = []
    for method_name, msum in summary.items():
        lines.append("")
        lines.append(f"######## METHOD: {method_name} ########")

        dump_pairs_epi(lines, method_name, msum)
        dump_epi_0N(lines, method_name, msum)
        dump_epi_0K(lines, method_name, msum)
        dump_epi_equalized_0N(lines, method_name, msum)
        dump_flops_time(lines, method_name, msum)

    args.out_txt.write_text("\n".join(lines), encoding="utf-8")
    print("✔ written:", args.out_txt)


if __name__ == "__main__":
    main()
