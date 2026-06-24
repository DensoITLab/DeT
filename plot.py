#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--json_path',
        type=Path,
        default=Path('/home/ach17765lb/data/phototourism/results_imc_eval_jamma_hist.json'),
    )

    parser.add_argument(
        '--save_dir',
        type=Path,
        default=Path('/home/ach17765lb/data/phototourism/jamma_assumption_hist_plots'),
    )

    parser.add_argument(
        '--crop_radius',
        type=float,
        default=5.0,
        help='Half crop size κ/2',
    )

    parser.add_argument(
        '--max_x',
        type=float,
        default=10.0,
    )

    parser.add_argument(
        '--bins',
        type=int,
        default=50,
    )

    return parser.parse_args()


def plot_all_radius_lines(
    data,
    radii,
    save_path,
    crop_radius=5.0,
    max_x=20.0,
    bins=28,
):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b",
    ]

    for idx, radius in enumerate(radii):
        errs = []
        for pair_res in data["per_pair"]:
            errs.extend(pair_res["radii"][radius]["errors_px"])

        errs = np.asarray(errs, dtype=np.float32)
        errs = errs[np.isfinite(errs)]

        n_total = len(errs)
        n_inside = np.sum(errs <= crop_radius)
        inside_pct = 100.0 * n_inside / n_total if n_total > 0 else 0.0

        counts, edges = np.histogram(
            errs,
            bins=bins,
            range=(0, max_x),
        )
        centers = 0.5 * (edges[:-1] + edges[1:])

        radius_f = float(radius)

        if np.isclose(radius_f, 0.0):
            color = "black"
            linestyle = "--"
            linewidth = 1.5
            zorder = 10
        else:
            color = colors[idx % len(colors)]
            linestyle = "-"
            linewidth = 1.2
            zorder = 5

        ax.plot(
            centers,
            counts,
            linestyle=linestyle,
            linewidth=linewidth,
            color=color,
            label=rf"$|p_A-q_A|={radius_f:.2f}px$, in={inside_pct:.1f}%",
            zorder=zorder,
        )

    ax.axvline(
        crop_radius,
        color="black",
        linestyle=":",
        linewidth=2.2,
    )

    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.15)


    ax.text(
        crop_radius * 0.7,
        ymax * 1.05,
        r"GT inside the $P$",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    # inside arrow
    ax.annotate(
        "",
        xy=(3.5, ymax * 0.999),
        xytext=(crop_radius - 0.2, ymax * 0.999),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.8,
            color="black",
        ),
    )

    ax.text(
        crop_radius + (max_x - crop_radius) * 0.33,
        ymax * 1.05,
        r"GT outside the $P$",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    # outside arrow
    ax.annotate(
        "",
        xy=(crop_radius + 0.2, ymax * 0.999),
        xytext=(max_x - 3.5, ymax * 0.999),
        arrowprops=dict(
            arrowstyle="<-",
            linewidth=0.8,
            color="black",
        ),
    )

    ax.set_xlabel(
        r"$MSE(q^{pseudo}-q_B^{GT})$",
        fontsize=9,
        labelpad=2,)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.set_yticks([])
    ax.set_xlim(0, max_x)

    ax.legend(
        frameon=False,
        fontsize=6.5,
        loc="center right",
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def plot_all_radius_bars(
    data,
    radii,
    save_path,
    crop_radius=5.0,
    max_x=20.0,
    bins=28,
):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]

    for idx, radius in enumerate(radii):

        errs = []
        for pair_res in data["per_pair"]:
            errs.extend(pair_res["radii"][radius]["errors_px"])

        errs = np.asarray(errs, dtype=np.float32)
        errs = errs[np.isfinite(errs)]

        counts, edges = np.histogram(
            errs,
            bins=bins,
            range=(0, max_x),
        )

        widths = edges[1:] - edges[:-1]
        centers = 0.5 * (edges[:-1] + edges[1:])

        radius_f = float(radius)

        n_total = len(errs)
        n_inside = np.sum(errs <= crop_radius)
        inside_pct = 100.0 * n_inside / n_total if n_total > 0 else 0.0

        # ===== style =====
        if np.isclose(radius_f, 0.0):
            color = "white"
            edgecolor = "black"
            linewidth = 0.7
            alpha = 1.0
            zorder = 5
        else:
            color = colors[idx % len(colors)]
            edgecolor = color
            linewidth = 0.5
            alpha = 0.22
            zorder = 10

        ax.bar(
            centers,
            counts,
            width=widths,
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            align="center",
            #label=rf"$|p_A-q_A|={radius_f:.2f}px$, in={inside_pct:.1f}%",
            label=" ",
            zorder=zorder,
        )
        print(rf"$|p_A-q_A|={radius_f:.2f}px$, in={inside_pct:.1f}%")
    # ===== crop boundary =====
    ax.axvline(
        crop_radius,
        color="black",
        linestyle=":",
        linewidth=2.2,
    )

    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.15)


    """
    ax.text(
        crop_radius * 0.7,
        ymax * 1.05,
        r"GT inside the $P$",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    # inside arrow
    ax.annotate(
        "",
        xy=(3.5, ymax * 0.999),
        xytext=(crop_radius - 0.2, ymax * 0.999),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.8,
            color="black",
        ),
    )

    ax.text(
        crop_radius + (max_x - crop_radius) * 0.33,
        ymax * 1.05,
        r"GT outside the $P$",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    # outside arrow
    ax.annotate(
        "",
        xy=(crop_radius + 0.2, ymax * 0.999),
        xytext=(max_x - 3.5, ymax * 0.999),
        arrowprops=dict(
            arrowstyle="<-",
            linewidth=0.8,
            color="black",
        ),
    )
    """
    #ax.set_xlabel(
    #    r"$MSE(q^{pseudo}-q_B^{GT})$",
    #    fontsize=9,
    #    labelpad=2,
    #)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.set_yticks([])
    ax.set_xticks([])

    ax.set_xlim(0, max_x)

    ax.legend(
        frameon=False,
        fontsize=6.5,
        loc="center",
        bbox_to_anchor=(0.55, 0.48)
    )

    fig.tight_layout(pad=0.3)

    fig.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

def main():
    args = parse_args()

    with open(args.json_path, 'r') as f:
        data = json.load(f)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    summary = data['summary']
    radii = sorted(summary['by_radius'].keys(), key=float)

    for radius in radii:
        errs = []
        for pair_res in data["per_pair"]:
            errs.extend(pair_res["radii"][radius]["errors_px"])

    all_line_path = args.save_dir / "assumption_all_lines.png"

    plot_all_radius_lines(
        data=data,
        radii=radii,
        save_path=all_line_path,
        crop_radius=args.crop_radius,
        max_x=args.max_x,
        bins=args.bins,
    )

    plot_all_radius_bars(
        data=data,
        radii=radii,
        save_path=args.save_dir / "assumption_all_bars.png",
        crop_radius=args.crop_radius,
        max_x=args.max_x,
        bins=args.bins,
    )

    print(f"[OK] saved: {all_line_path}")


if __name__ == '__main__':
    main()