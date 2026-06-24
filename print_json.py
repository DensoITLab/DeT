#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import argparse


def dump_0N_tracks_one_side(lines, method_name, msum):
    info = msum.get("depth_0N_tracks_summary_one_side")
    if info is None:
        return

    thr_list = info.get("thr_px", [])
    counts = info.get("counts", [])
    ratios = info.get("ratios", [])
    n_tracks = int(info.get("n_tracks_total_0N", 0))
    median_px = float(info.get("median_err_px_valid_only", float("nan")))

    lines.append("[Depth 0→N (one-sided tracks)]")
    lines.append(f"  n_tracks_total_0N        : {n_tracks}")
    lines.append(f"  median_err_px_valid_only : {median_px:.6f}")
    lines.append("  thresholds:")
    for thr, c, r in zip(thr_list, counts, ratios):
        lines.append(
            f"    thr = {int(thr):2d} px:"
            f"  count = {int(c):8d},  ratio = {float(r):.6f}"
        )
    lines.append("")  # 空行


def dump_0N_tracks_sym(lines, method_name, msum):
    info = msum.get("depth_0N_tracks_summary_sym")
    if info is None:
        return

    thr_list = info.get("thr_px", [])
    counts = info.get("counts", [])
    ratios = info.get("ratios", [])
    n_tracks = int(info.get("n_tracks_total_0N", 0))
    median_px = float(info.get("median_err_px_valid_only", float("nan")))

    lines.append("[Depth 0↔N (symmetric tracks)]")
    lines.append(f"  n_tracks_total_0N        : {n_tracks}")
    lines.append(f"  median_err_px_valid_only : {median_px:.6f}")
    lines.append("  thresholds:")
    for thr, c, r in zip(thr_list, counts, ratios):
        lines.append(
            f"    thr = {int(thr):2d} px:"
            f"  count = {int(c):8d},  ratio = {float(r):.6f}"
        )
    lines.append("")


def dump_0N_summary(lines, method_name, msum):
    info = msum.get("0N_summary")
    if info is None:
        return

    auc5 = float(info.get("auc_0N@5", float("nan")))
    auc10 = float(info.get("auc_0N@10", float("nan")))
    auc20 = float(info.get("auc_0N@20", float("nan")))
    prec1 = float(info.get("global_precision_0N@1e-4_by_counts", 0.0))
    prec5 = float(info.get("global_precision_0N@5e-4_by_counts", 0.0))

    gc = info.get("global_counts_0N", {})
    c1 = int(gc.get("correct_1e-4", 0))
    c5 = int(gc.get("correct_5e-4", 0))
    total = int(gc.get("total", 0))

    ts = info.get("global_track_survival_0_to_N", {})
    tracks_full = int(ts.get("tracks_full_0_to_N", 0))
    tracks_start = int(ts.get("tracks_start_at_0", 0))
    frac_full = float(ts.get("fraction_full_0_to_N", 0.0))

    bags_valid = int(info.get("bags_with_valid_pose_0N", 0))
    bags_total = int(info.get("bags_total", 0))

    lines.append("[Epipolar 0→N summary]")
    lines.append(f"  AUC@5°  : {auc5:.2f}")
    lines.append(f"  AUC@10° : {auc10:.2f}")
    lines.append(f"  AUC@20° : {auc20:.2f}")
    lines.append(
        f"  Precision@1e-4: {prec1:.6f}"
        f"  ({c1} / {total})"
    )
    lines.append(
        f"  Precision@5e-4: {prec5:.6f}"
        f"  ({c5} / {total})"
    )
    lines.append(
        f"  Tracks full 0→N: {tracks_full} / {tracks_start}"
        f"  (fraction = {frac_full:.6f})"
    )
    lines.append(
        f"  Bags with valid pose: {bags_valid} / {bags_total}"
    )
    lines.append("")


def dump_flops_time(lines, method_name, msum):
    info = msum.get("flops_time_summary")
    if info is None:
        return

    total_pairs = int(info.get("total_pairs", 0))
    total_flops = float(info.get("total_flops_GMac", 0.0))
    total_runtime = float(info.get("total_runtime_ms", 0.0))
    avg_flops = float(info.get("avg_flops_per_pair_GMac", 0.0))
    avg_time = float(info.get("avg_runtime_per_pair_ms", 0.0))

    lines.append("[FLOPs / runtime summary]")
    lines.append(f"  total_pairs              : {total_pairs}")
    lines.append(f"  total_flops_GMac         : {total_flops:.6f}")
    lines.append(f"  total_runtime_ms         : {total_runtime:.6f}")
    lines.append(f"  avg_flops_per_pair_GMac  : {avg_flops:.6f}")
    lines.append(f"  avg_runtime_per_pair_ms  : {avg_time:.6f}")
    lines.append("")


# ============================================================
# 追加: imgN (=k) ごとの depth (symmetric) カウントを吐く
# ============================================================

def dump_depth_0k_sym_counts(lines, method_name, msum, thr_px_target=3):
    """
    修正版:
    depth_0k_summary_sym の track_counts_survival_global から
    「0↔k で、0→k の全ステップで一度も閾値を破っていないトラック数」
    (survival count) を出す。

    これなら基本 k が増えるほど減る挙動になる。
    """
    info = msum.get("depth_0k_summary_sym")
    if info is None:
        return

    k_list = sorted([int(k) for k in info.keys()], key=int)
    if not k_list:
        return

    # thr のキーは "1","3","5","10" みたいな文字列で入ってる前提
    thr_key = str(int(thr_px_target))

    lines.append(f"[Depth 0↔k symmetric SURVIVAL tracks @ {thr_px_target}px]")
    lines.append("  format: k: survival_tracks  (note: survival over edges 0→1..k-1→k)")
    for k in k_list:
        rec = info.get(str(k), {})
        tcs = rec.get("track_counts_survival_global", {})  # { "1":..., "3":..., ... }

        c = tcs.get(thr_key, 0)
        # 念のため int 化
        try:
            c = int(c)
        except Exception:
            c = 0

        lines.append(f"    k = {k:2d}: {c:8d}")
    lines.append("")


def dump_depth_gt_visibility_counts(lines, method_name, msum, thr_px_target=3):
    """
    JamMa のみ存在する depth_gt_visibility_from0_global から、
    GT（両側）として multi_thr の total_counts_per_k を出力する。
    """
    vis = msum.get("depth_gt_visibility_from0_global")
    if vis is None:
        return

    multi = vis.get("multi_thr", None)
    if multi is None:
        return

    thr_list = multi.get("thr_px", [])
    counts_per_k = multi.get("total_counts_per_k", [])  # shape (T, bag_size)

    if not thr_list or not counts_per_k:
        return

    if thr_px_target in thr_list:
        tidx = thr_list.index(thr_px_target)
    else:
        tidx = 1  # fallback 3px想定

    seq = counts_per_k[tidx] if tidx < len(counts_per_k) else []
    total_init = int(vis.get("total_initial_points", 0))

    lines.append(f"[GT visibility (symmetric depth) @ {thr_px_target}px]")
    lines.append(f"  total_initial_points: {total_init}")
    lines.append("  format: frame_k: count (ratio vs initial)")
    for k, c in enumerate(seq):
        c = int(c)
        ratio = (c / total_init) if total_init > 0 else 0.0
        lines.append(f"    k = {k:2d}: {c:8d}  (ratio={ratio:.6f})")
    lines.append("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=Path("/home/ach17765lb/data/phototourism/results_imc_eval_summary_splg.json"),
        help="results_megadepth_eval_summary.json (summary のみ) のパス",
    )
    parser.add_argument(
        "--out_txt",
        type=Path,
        default=Path("/home/ach17765lb/data/phototourism/results_imc_eval_depth_summary_splg_st_peters_square.txt"),
        help="出力する txt のパス",
    )
    args = parser.parse_args()

    summary = json.loads(args.summary_json.read_text())

    lines = []

    lines.append(f"Scene: {summary.get('scene_name', 'unknown')}")
    lines.append("")

    for method_name, msum in summary.items():
        if method_name == "scene_name":
            continue
        lines.append("")
        lines.append(f"########## METHOD: {method_name} ##########")
        lines.append("")

        dump_0N_tracks_one_side(lines, method_name, msum)
        dump_0N_tracks_sym(lines, method_name, msum)
        dump_0N_summary(lines, method_name, msum)
        dump_flops_time(lines, method_name, msum)

        # --- 追加: imgN (=k) ごとの depth（symmetric）3px count ---
        dump_depth_0k_sym_counts(lines, method_name, msum, thr_px_target=3)

        # --- 追加: GT（両側）3px 生存数（JamMaのみ。無ければ何も出さない） ---
        dump_depth_gt_visibility_counts(lines, method_name, msum, thr_px_target=3)

    args.out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"✔ Output written to: {args.out_txt}")


if __name__ == "__main__":
    main()