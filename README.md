# DeT: Detect by Track

[![license](https://img.shields.io/badge/LICENSE-MIT-green)](LICENSE)
[![conference](https://img.shields.io/badge/ECCV-2026-blue)](https://eccv.ecva.net/)
[![base](https://img.shields.io/badge/base-JamMa-6f42c1)](https://github.com/leoluxxx/JamMa)

Official PyTorch implementation of **Detect by Track: Making Detector-Free Matcher Trackable**.

Detector-free matchers such as EDM and JamMa provide strong pairwise matching accuracy and runtime, but their implicit keypoint selection can fragment tracks across multiple views. DeT directly steers the local similarity matrix from an arbitrary sub-pixel query location, allowing a detector-free matcher to produce connected tracks while preserving its matching-oriented keypoint selection.

This release implements DeT on top of [JamMa](https://github.com/leoluxxx/JamMa). Some configuration keys and internal module names still use `jamma` to stay compatible with the upstream code structure.

## Highlights

- DeT/JamMa pair matching demo.
- MegaDepth and ScanNet two-view evaluation scripts.
- IMC/SfM and multi-view tracking evaluation scripts.
- MegaDepth training entry point and data setup notes.
- Checkpoints, datasets, logs, and generated evaluation files are intentionally excluded from Git.

## Installation

```bash
conda env create -f environment.yaml
conda activate det
pip install -r requirements.txt
pip install mamba-ssm==2.0.3
```

`mamba-ssm` depends on the local CUDA/PyTorch setup. Installing a wheel that matches your CUDA version is usually faster and more reliable than building from source.

## Checkpoints

Checkpoint files are not committed to this repository. Download the DeT checkpoint separately and pass its path with `--ckpt_path` or `--jamma_ckpt`, depending on the script.

```bash
python demo/demo.py --ckpt_path /path/to/det.ckpt
python test.py configs/data/megadepth_test_1500.py configs/jamma/outdoor/test.py --ckpt_path /path/to/det.ckpt
```

For compatibility checks, scripts that accept `--ckpt_path official` load the upstream JamMa checkpoint from the JamMa release.

## Demo

```bash
python demo/demo.py \
  --image1 /path/to/image0.jpg \
  --image2 /path/to/image1.jpg \
  --ckpt_path /path/to/det.ckpt \
  --output_dir demo/output
```

## Data

MegaDepth and ScanNet follow the same data organization as JamMa. See [docs/TRAINING.md](docs/TRAINING.md) for dataset and index setup. IMC/SfM and tracking scripts expect bag files, images, calibration, and depth maps under `data/`.

## Evaluation

Edit checkpoint and data paths inside the scripts, then run:

```bash
bash scripts/reproduce_test/outdoor.sh
bash scripts/reproduce_test/indoor.sh
bash scripts/reproduce_test/imc_sfm.sh
bash scripts/reproduce_test/imc_tracking.sh
bash scripts/reproduce_test/megadepth_tracking.sh
```

The two-view entry point is `test.py`. The online SfM benchmark is `test_imc_sfm.py`. The multi-view tracking benchmarks are `test_imc_tracking.py` and `test_megadepth_tracking.py`; they write summaries under `outputs/`.

## Training

```bash
bash scripts/reproduce_train/outdoor.sh
```

The training script writes TensorBoard logs and checkpoints under `det_log/`.

## Repository Layout

```text
configs/                   Configuration files
demo/                      Pair matching demo
docs/                      Dataset setup notes
scripts/reproduce_test/    Evaluation entry points
scripts/reproduce_train/   Training entry points
src/                       Model, data, losses, and utilities
test.py                    MegaDepth/ScanNet two-view evaluation
test_imc_sfm.py            IMC online SfM evaluation
test_imc_tracking.py       IMC multi-view tracking evaluation
test_megadepth_tracking.py MegaDepth multi-view tracking evaluation
train.py                   MegaDepth training entry point
```

## Acknowledgements

This codebase is based on [JamMa](https://github.com/leoluxxx/JamMa). Parts of the upstream code are derived from LoFTR and XoFTR. We thank the authors for releasing their implementations.

## Citation

The camera-ready citation will be updated once the official metadata is public.

```bibtex
@inproceedings{det2026,
  title     = {Detect by Track: Making Detector-Free Matcher Trackable},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
