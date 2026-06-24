# DeT

Official PyTorch implementation of **DeT**.

> TODO before public release: replace this line with the full paper title, venue, paper/arXiv link, project page, and final citation.

This repository is built on top of [JamMa](https://github.com/leoluxxx/JamMa). Configuration keys and some internal module names still use `jamma` to keep compatibility with the upstream code structure.

## Highlights

- Training and evaluation code for DeT.
- Demo script for matching an image pair.
- Reproduction scripts for MegaDepth and ScanNet style evaluation.
- Public release hygiene: checkpoints, datasets, logs, and generated files are intentionally not tracked by Git.

## Installation

```bash
conda env create -f environment.yaml
conda activate det
pip install -r requirements.txt
pip install mamba-ssm==2.0.3
```

`mamba-ssm` depends on the local CUDA/PyTorch setup. Installing a wheel that matches your CUDA version is usually faster and more reliable than building from source.

## Checkpoints

Checkpoints are not committed to this repository. Put downloaded weights under `weights/` and pass the path with `--ckpt_path`.

```text
weights/
  det.ckpt
```

> TODO before public release: upload the DeT checkpoint to GitHub Releases, Hugging Face, or another stable artifact host, then add the download URL here.

For compatibility experiments, scripts that accept `--ckpt_path official` load the upstream JamMa checkpoint from the JamMa release.

## Demo

```bash
python demo/demo.py \
  --image1 /path/to/image0.jpg \
  --image2 /path/to/image1.jpg \
  --ckpt_path weights/det.ckpt \
  --output_dir demo/output
```

If you only want to verify that the upstream pipeline runs, use:

```bash
python demo/demo.py --ckpt_path official
```

## Evaluation

Prepare the testing subsets and dataset indices as described in [docs/TRAINING.md](docs/TRAINING.md), then run:

```bash
bash scripts/reproduce_test/outdoor.sh
bash scripts/reproduce_test/indoor.sh
```

Edit `ckpt_path` in the script or pass a checkpoint path directly to `test.py`.

## Training

Follow [docs/TRAINING.md](docs/TRAINING.md) to prepare MegaDepth and ScanNet. Then run:

```bash
bash scripts/reproduce_train/outdoor.sh
```

Training logs and checkpoints are written to local output directories and are ignored by Git.

## Repository Layout

```text
configs/                 Configuration files
demo/                    Pair matching demo
docs/                    Dataset and training notes
scripts/reproduce_test/  Evaluation entry points
scripts/reproduce_train/ Training entry points
src/                     Model, data, loss, and utility code
train.py                 Training entry point
test.py                  Evaluation entry point
```

## Acknowledgements

This codebase is based on [JamMa](https://github.com/leoluxxx/JamMa). We thank the JamMa authors for releasing their code.

## Citation

> TODO before public release: replace this placeholder with the accepted paper BibTeX.

```bibtex
@inproceedings{det2026,
  title     = {TODO: Full DeT paper title},
  author    = {TODO: Author list},
  booktitle = {TODO: Venue},
  year      = {2026}
}
```
