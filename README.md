# DeT: Detect by Track

Official PyTorch implementation of **Detect by Track: Making Detector-Free Matcher Trackable**.

DeT makes detector-free matchers trackable by steering the local similarity matrix from a previous sub-pixel track location. This repository provides the DeT implementation built on top of [JamMa](https://github.com/leoluxxx/JamMa). Configuration keys and some internal module names still use `jamma` for compatibility with the upstream code structure.

## Highlights

- Pair matching demo.
- MegaDepth, IMC/SfM, and Fig. 6 tracking benchmark scripts.
- Inference-time DeT modification for JamMa-based matching.
- Checkpoints, datasets, logs, and generated files are intentionally not tracked by Git.

## Installation

```bash
conda env create -f environment.yaml
conda activate det
pip install -r requirements.txt
pip install mamba-ssm==2.0.3
```

`mamba-ssm` depends on the local CUDA/PyTorch setup. Installing a wheel that matches your CUDA version is usually faster and more reliable than building from source.

## Checkpoints

Checkpoint files are not committed to this repository. Download the DeT checkpoint separately and pass its path with `--ckpt_path`.

```bash
python demo/demo.py --ckpt_path /path/to/det.ckpt
python test.py configs/data/megadepth_test_1500.py configs/jamma/outdoor/test.py --ckpt_path /path/to/det.ckpt
```

The DeT checkpoint will be provided as a separate release artifact. After downloading it, pass the checkpoint path with `--ckpt_path` or `--jamma_ckpt`, depending on the script.

For compatibility experiments, scripts that accept `--ckpt_path official` load the upstream JamMa checkpoint from the JamMa release.

## Demo

```bash
python demo/demo.py \
  --image1 /path/to/image0.jpg \
  --image2 /path/to/image1.jpg \
  --ckpt_path /path/to/det.ckpt \
  --output_dir demo/output
```

If you only want to verify that the upstream pipeline runs, use:

```bash
python demo/demo.py --ckpt_path official
```

## Evaluation

Prepare the MegaDepth testing subset or IMC bag files/calibration, then run:

```bash
bash scripts/reproduce_test/outdoor.sh
bash scripts/reproduce_test/imc_sfm.sh
bash scripts/reproduce_test/imc_tracking.sh
bash scripts/reproduce_test/megadepth_tracking.sh
```

Edit the checkpoint path in the script, or pass it directly with `--ckpt_path` for MegaDepth and Fig. 6 tracking evaluation, and `--jamma_ckpt` for IMC/SfM evaluation. The Fig. 6 tracking benchmark is produced by `test_imc_tracking.py` and `test_megadepth_tracking.py`; these scripts write summary JSON files under `outputs/`.

## Repository Layout

```text
configs/                 Configuration files
demo/                    Pair matching demo
scripts/reproduce_test/  Evaluation entry points
src/                     Model, data, and utility code
test.py                  MegaDepth evaluation entry point
test_imc_sfm.py          IMC/SfM evaluation entry point
test_imc_tracking.py     IMC tracking benchmark for Fig. 6
test_megadepth_tracking.py MegaDepth tracking benchmark for Fig. 6
```

## Acknowledgements

This codebase is based on [JamMa](https://github.com/leoluxxx/JamMa). We thank the JamMa authors for releasing their code.

## Citation

```bibtex
@inproceedings{det2026,
  title     = {Detect by Track: Making Detector-Free Matcher Trackable},
  author    = {DeT Authors},
  booktitle = {ECCV},
  year      = {2026}
}
```
