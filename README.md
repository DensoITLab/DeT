# DeT: Detect by Track

[![license](https://img.shields.io/badge/LICENSE-MIT-green)](LICENSE)
[![conference](https://img.shields.io/badge/ECCV-2026-blue)](https://eccv.ecva.net/)
[![base](https://img.shields.io/badge/base-JamMa-6f42c1)](https://github.com/leoluxxx/JamMa)

Official PyTorch implementation of **Detect by Track: Making Detector-Free Matcher Trackable**.

DeT makes detector-free matching trackable across multiple views by steering the local similarity matrix from arbitrary sub-pixel query locations. This release implements DeT on top of [JamMa](https://github.com/leoluxxx/JamMa) and is intentionally scoped to:

- DeT sequence demo with three or more ordered images.
- Paper evaluation launch scripts for IMC and MegaDepth scenes 0015/0022.
- IMC/MegaDepth multi-view tracking and online SfM evaluation.

Checkpoints, datasets, logs, and generated evaluation files are intentionally excluded from Git. Some internal module names and config keys still use `jamma` for compatibility with the upstream JamMa code structure.
Only the upstream JamMa files needed for DeT inference and evaluation are kept in the tree; the public DeT demo entry point is `demo/demo_det.py`.

## Installation

```bash
conda env create -f environment.yaml
conda activate det
pip install -r requirements.txt
pip install mamba-ssm==2.0.3
```

`mamba-ssm` depends on the local CUDA/PyTorch setup. Installing a wheel that matches your CUDA version is usually faster than building from source.

## Checkpoints

Download the DeT checkpoint separately and place it at `weights/jamma.ckpt` for the default demo and tracking evaluation:

```bash
python demo/demo_det.py
```

By default, the demo reads `weights/jamma.ckpt` and the first three Piazza San Marco images under `assets/phototourism_sample_images`.
The paper tracking evaluation scripts use `--ckpt_path`:

```bash
python -m eval.eval_imc --ckpt_path weights/jamma.ckpt
python -m eval.eval_megadepth --scene_name 0015 --ckpt_path weights/jamma.ckpt
python -m eval.eval_megadepth --scene_name 0022 --ckpt_path weights/jamma.ckpt
```

## DeT Demo

The demo compares NN-JamMa and DeT-JamMa tracking on the same ordered image sequence. With the default sample files and checkpoint in place, it runs without arguments:

```bash
python demo/demo_det.py
```

It also accepts explicit image paths:

```bash
python demo/demo_det.py \
  --images /path/to/image0.jpg /path/to/image1.jpg /path/to/image2.jpg \
  --ckpt_path /path/to/det.ckpt \
  --output_dir demo/output_det
```

You can also pass a directory:

```bash
python demo/demo_det.py \
  --image_dir /path/to/sequence \
  --pattern "*.jpg" \
  --ckpt_path /path/to/det.ckpt
```

The demo writes `tracks.json` and `comparison.png` under the output directory. The comparison figure shows NN-JamMa on top and DeT-JamMa on the bottom, using the same sampled start tracks from the shared 0-1 matches. It adds one shared zoom window after the third image for a seeded random common track, with the NN point in green and the DeT point in red.
The default visualization uses a 720 px row height. Use `--viz_height`, `--label_font_size`, `--point_radius`, `--point_alpha`, `--zoom_window_size`, `--zoom_crop_size`, `--zoom_seed`, and `--max_viz_tracks` to adjust the comparison figure.

## Paper Evaluation

The paper evaluation scripts run 5-frame bags for IMC and MegaDepth.

```bash
python -m eval.eval_imc --ckpt_path weights/jamma.ckpt
python -m eval.eval_megadepth --scene_name 0015 --ckpt_path weights/jamma.ckpt
python -m eval.eval_megadepth --scene_name 0022 --ckpt_path weights/jamma.ckpt
```

The evaluation entry points are:

- `eval/eval_imc.py`: IMC tracking metrics for the left and middle plots in Fig. 6.
- `eval/eval_megadepth.py`: MegaDepth tracking metrics for the left and middle plots in Fig. 6.

Default data locations are `data/imc` and `data/megadepth`. Override them with `--imc_root`, `--megadepth_root`, `--megadepth_sfm_root`, `--subset_dir`, `--dataset_root`, or `--calib_dir`. Results are written under `outputs/fig6_imc_tracking*.json` and `outputs/fig6_megadepth_tracking*.json`.
The MegaDepth script also accepts `15` and `22` and normalizes them to `0015` and `0022`.
The tracking scripts run `--methods nn-jamma det-jamma` by default. They report average correct tracks from frame 1 to frame 5 using symmetric epipolar error `< 1e-3`, average FLOPs per pair, and average model inference time per pair. FLOPs are profiled once per method state and reused instead of being measured for every image pair. The timing window covers only the model forward pass, not image loading, preprocessing, track linking, epipolar scoring, FLOPs profiling, or warmup.

To add another tracker, add one decorated pair-matching function to `eval/matchers.py` that returns `PairMatchOutput`. The evaluator handles bag loading, track linking, epipolar scoring, FLOPs/time aggregation, and JSON output.

## Repository Layout

```text
configs/                   Inference and image preprocessing configs
demo/demo_det.py           DeT sequence demo for 3+ images
eval/eval_imc.py           IMC tracking metrics
eval/eval_megadepth.py     MegaDepth tracking metrics
eval/matchers.py           Pair-matching registry and model adapters
eval/tracking.py           Dataset-agnostic tracking evaluation loop
eval/geometry.py           Camera loading and epipolar scoring
eval/options.py            Shared tracking-evaluation CLI options
eval/records.py            Evaluation data containers
src/                       DeT/JamMa model and utility code
```

## Acknowledgements

This codebase is based on [JamMa](https://github.com/leoluxxx/JamMa). Parts of the upstream code are derived from LoFTR and XoFTR. We thank the authors for releasing their implementations.
The `assets/` directory is copied from the upstream JamMa repository and is used for the default demo inputs.

## Citation

The camera-ready citation will be updated once the official metadata is public.

```bibtex
@inproceedings{det2026,
  title     = {Detect by Track: Making Detector-Free Matcher Trackable},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
