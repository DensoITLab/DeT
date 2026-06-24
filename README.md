# DeT

Official PyTorch implementation of **DeT**.

> TODO before public release: replace this line with the full paper title, venue, paper/arXiv link, project page, and final citation.

This repository is built on top of [JamMa](https://github.com/leoluxxx/JamMa). Configuration keys and some internal module names still use `jamma` to keep compatibility with the upstream code structure.

## Highlights

- Evaluation code for DeT.
- Demo script for matching an image pair.
- Reproduction scripts for MegaDepth and ScanNet style evaluation.
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

> TODO before public release: upload the DeT checkpoint to GitHub Releases, Hugging Face, or another stable artifact host, then add the download URL here.

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

Prepare the testing subsets and dataset indices, then run:

```bash
bash scripts/reproduce_test/outdoor.sh
bash scripts/reproduce_test/indoor.sh
```

Edit `ckpt_path` in the script or pass a checkpoint path directly to `test.py`.

## Repository Layout

```text
configs/                 Configuration files
demo/                    Pair matching demo
docs/                    Dataset setup notes
scripts/reproduce_test/  Evaluation entry points
src/                     Model, data, loss, and utility code
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
