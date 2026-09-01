# Detect by Track: Making Detector-Free Matcher Trackable

[![license](https://img.shields.io/badge/LICENSE-MIT-green)](LICENSE)
![conference](https://img.shields.io/badge/ECCV-2026-blue)
[![base](https://img.shields.io/badge/base-JamMa-6f42c1)](https://github.com/leoluxxx/JamMa)

Official PyTorch implementation of **Detect by Track: Making Detector-Free Matcher Trackable**.

## Installation

```bash
conda env create -f environment.yaml
conda activate det
pip install -r requirements.txt
pip install mamba-ssm==2.0.3
```

## Checkpoint

Place the checkpoint at `weights/jamma.ckpt`.

## Demo

```bash
python demo/demo_det.py
```

The default demo uses the first three Piazza San Marco images in `assets/phototourism_sample_images` and writes results to `demo/output_det`.

## Evaluation

```bash
python -m eval.eval_imc --ckpt_path weights/jamma.ckpt
python -m eval.eval_megadepth --scene_name 0015 --ckpt_path weights/jamma.ckpt
python -m eval.eval_megadepth --scene_name 0022 --ckpt_path weights/jamma.ckpt
```

The evaluation scripts run NN-JamMa and DeT-JamMa by default. Add a new method by registering one pair-matching function in `eval/eval_utils.py`.

## Acknowledgements

This codebase is based on [JamMa](https://github.com/leoluxxx/JamMa). Parts of the upstream code are derived from LoFTR and XoFTR.

## Citation

```bibtex
@inproceedings{det2026,
  author    = {Yusuke Sekikawa and Hideki Shirai and Ruka Eto and Yuzhe Hao and Kengo Mitsui and Nakamasa Inoue},
  title     = {Detect by Track: Making Detector-Free Matcher Trackable},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
