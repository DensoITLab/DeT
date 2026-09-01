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
The IMC SfM script uses `--jamma_ckpt`, and the tracking scripts use `--ckpt_path`.
The paper evaluation scripts use `CKPT_PATH`:

```bash
CKPT_PATH=/path/to/det.ckpt bash scripts/reproduce_test/paper_all.sh
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

The paper evaluation scripts run 5-frame bags for IMC and MegaDepth. Set the checkpoint path and, if needed, override dataset roots with environment variables.

```bash
CKPT_PATH=/path/to/det.ckpt bash scripts/reproduce_test/paper_imc.sh
CKPT_PATH=/path/to/det.ckpt bash scripts/reproduce_test/paper_megadepth_0015_0022.sh
CKPT_PATH=/path/to/det.ckpt bash scripts/reproduce_test/paper_all.sh
```

The evaluation entry points are:

- `scripts/reproduce_test/paper_imc.sh`: IMC `reichstag`, `sacre_coeur`, and `st_peters_square`.
- `scripts/reproduce_test/paper_megadepth_0015_0022.sh`: MegaDepth `0015` and `0022`.
- `eval_imc.py`: IMC tracking metrics for the left and middle plots in Fig. 6.
- `eval_megadepth.py`: MegaDepth tracking metrics for the left and middle plots in Fig. 6.
- `test_imc_sfm.py`: online SfM evaluation for `--method det-jamma` and `--method nn-jamma`.

Default data locations are `data/imc` and `data/megadepth`. Override them with `IMC_ROOT`, `MEGADEPTH_ROOT`, or `MEGADEPTH_SFM_ROOT`. Results are written under `outputs/paper/imc` and `outputs/paper/megadepth`.
The MegaDepth script also accepts `15 22` and normalizes them to `0015 0022`.
The tracking scripts run `--methods nn-jamma det-jamma` by default. They report average correct tracks from frame 1 to frame 5 using symmetric epipolar error `< 1e-3`, average FLOPs per pair, and average model inference time per pair. FLOPs are profiled once per method state and reused instead of being measured for every image pair. The timing window covers only the model forward pass, not image loading, preprocessing, track linking, epipolar scoring, FLOPs profiling, or warmup.

To add another tracker, add one decorated pair-matching function to `eval_tracking_common.py` that returns `PairMatchOutput`. The evaluator handles bag loading, track linking, epipolar scoring, FLOPs/time aggregation, and JSON output.

## Repository Layout

```text
configs/                   Inference and image preprocessing configs
demo/demo_det.py           DeT sequence demo for 3+ images
eval_tracking_common.py    Shared Fig. 6 tracking evaluation utilities
eval_imc.py                IMC tracking metrics
eval_megadepth.py          MegaDepth tracking metrics
scripts/reproduce_test/    Paper evaluation launch scripts
src/                       DeT/JamMa model and utility code
test_imc_sfm.py            IMC-style online SfM evaluation
test_imc_tracking.py       Compatibility wrapper for eval_imc.py
test_megadepth_tracking.py Compatibility wrapper for eval_megadepth.py
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
