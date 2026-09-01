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

Download the DeT checkpoint separately and place it at `weights/jamma.ckpt` for the default demo:

```bash
python demo/demo_det.py
```

By default, the demo reads `weights/jamma.ckpt` and the London Bridge images under `assets/phototourism_sample_images`.
The IMC SfM script uses `--jamma_ckpt`, and the tracking scripts use `--ckpt_path`. Scripts that accept `official` load the upstream JamMa checkpoint for compatibility checks.
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

The demo writes `tracks.json` and `comparison.png` under the output directory. The comparison figure shows NN-JamMa on top and DeT-JamMa on the bottom, using the same sampled start tracks from the shared 0-1 matches.
The default visualization uses a 720 px row height. Use `--viz_height`, `--label_font_size`, `--point_radius`, `--point_alpha`, and `--max_viz_tracks` to adjust the comparison figure.

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
- `test_imc_tracking.py`: IMC multi-view DeT track evaluation with epipolar, depth visibility, runtime, and FLOPs summaries.
- `test_megadepth_tracking.py`: MegaDepth multi-view DeT track evaluation for 5-frame bags.
- `test_imc_sfm.py`: online SfM evaluation for `--method det-jamma` and `--method nn-jamma`.

Default data locations are `data/imc` and `data/megadepth`. Override them with `IMC_ROOT`, `MEGADEPTH_ROOT`, or `MEGADEPTH_SFM_ROOT`. Results are written under `outputs/paper/imc` and `outputs/paper/megadepth`.
The MegaDepth script also accepts `15 22` and normalizes them to `0015 0022`.
The tracking scripts accept `--methods jamma det-jamma jamma_legacy`; the wrapper scripts expose this as `METHODS`.

## Repository Layout

```text
configs/                   Inference and image preprocessing configs
demo/demo_det.py           DeT sequence demo for 3+ images
scripts/reproduce_test/    Paper evaluation launch scripts
src/                       DeT/JamMa model and utility code
test_imc_sfm.py            IMC-style online SfM evaluation
test_imc_tracking.py       IMC multi-view tracking evaluation
test_megadepth_tracking.py MegaDepth multi-view tracking evaluation
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
