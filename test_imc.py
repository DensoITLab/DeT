#!/usr/bin/env python3
"""
Run Jamma inference only on adjacent pairs from 5bag_* txt files
for all Phototourism scenes, using read_megadepth_color() style loading.

Usage:
  python run_jamma_phototourism_bag5_adjacent_megadepth.py \
      --phototourism_root datasets/phototourism \
      --out_root /data/jamma/phototourism
"""
import argparse
import pprint
from pathlib import Path
import torch
import torch.nn.functional as F
from save_jamma_pair_for_imc import save_jamma_pair_for_imc
from src.utils.dataset import read_megadepth_color
from src.lightning.lightning_jamma import PL_JamMa
import pytorch_lightning as pl
from loguru import logger as loguru_logger
from src.config.default import get_cfg_defaults
from src.utils.profiler import build_profiler

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_cfg_path', type=str, default="configs/data/megadepth_test_1500.py")
    parser.add_argument('--main_cfg_path', type=str, default="configs/jamma/outdoor/test.py")
    parser.add_argument('--ckpt_path', type=str, default="official")
    parser.add_argument('--dump_dir', type=str, default="dump/jamma_imc")
    parser.add_argument('--profiler_name', type=str, default="inference")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--thr', type=float, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def normalize_line(s: str) -> str:
    return Path(s.strip()).stem if s.strip() else ""

def load_megadepth(path, device):
    # Megadepth-style image + mask 読み込み
    image, scale, mask, prepad_size, origin_wh, new_wh = read_megadepth_color(str(path), 832, 8, True)
    mask = F.interpolate(mask[None, None].float(), scale_factor=0.125,
                         mode='nearest', recompute_scale_factor=False)[0].bool()
    return image.to(device), mask.to(device)

def img_path(img_dir: Path, stem: str):
    for ext in [".jpg", ".png", ".jpeg"]:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found for stem={stem}")

def run_scene(model, scene_dir: Path, out_scene: Path, device):
    set_dir = scene_dir / "set_100"
    img_dir = set_dir / "images"
    sub_dir = set_dir / "sub_set"
    out_dir = out_scene / scene_dir.name / "jamma"
    out_dir.mkdir(parents=True, exist_ok=True)

    txts = sorted(sub_dir.glob("5bag_*.txt"))
    if not txts:
        print(f"[skip] no 5bag_*.txt in {sub_dir}")
        return

    print(f"=== Scene {scene_dir.name}: {len(txts)} files matching 5bag_*.txt ===")
    # jamma = build_jamma(...).to(device).eval()  # ←ユーザ環境でロード

    for t in txts:
        lines = [normalize_line(l) for l in t.read_text().splitlines() if l.strip()]
        print(f"--- {t.name}: {len(lines)} images ---")
        if len(lines) < 2:
            continue
        for i in range(len(lines) - 1):  # 隣接ペアのみ
            a, b = lines[i], lines[i + 1]
            ia, ib = img_path(img_dir, a), img_path(img_dir, b)
            image0, mask0 = load_megadepth(ia, device)
            image1, mask1 = load_megadepth(ib, device)

            if i == 0:
                data_ij = {
                    'imagec_0': image0.to(device), 
                    'imagec_1': image1.to(device),
                    'mask0': mask0.to(device),
                    'mask1': mask1.to(device),
                    'algo_res': True
                }
            else:
                data_ij = {
                    'imagec_0': image0.to(device), 
                    'imagec_1': image1.to(device),
                    'mask0': mask0.to(device),
                    'mask1': mask1.to(device),
                    'algo_res': True,
                    'prev_data': data_ij
                }

            model(data_ij)  # ←ここで推論を実行

            if 'mkpts0_f_origin' not in data_ij:
                data_ij['mkpts0_f_origin'] = torch.zeros(0,2, device=device)
                data_ij['mkpts1_f_origin'] = torch.zeros(0,2, device=device)

            save_jamma_pair_for_imc(data_ij, a, b, out_dir, 2048, scene_dir.name)


def main():
    # parse arguments
    args = parse_args()
    args.gpus = 1
    args.accelerator = 'gpu'
    args.benchmark = True
    print(args)
    pprint.pprint(vars(args))

    device = 'cuda'

    # === 固定パス ===
    root = Path("/home/ach17765lb/data/phototourism")        # ← Phototourism データセットのルート
    out_root = Path("/home/ach17765lb/JamMa/outputs")       # ← 出力保存先
    # =================

        # cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)

    if args.thr is not None:
        config.LOFTR.MATCH_COARSE.THR = args.thr

    loguru_logger.info("Args and config initialized!")

    # model
    profiler = build_profiler(args.profiler_name)
    model = PL_JamMa(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, dump_dir=args.dump_dir)
    model = model.to(device)
    loguru_logger.info("JamMa-lightning initialized!")

    scenes = ["reichstag", "sacre_coeur", "st_peters_square"]
    scenes = ['reichstag']
    for s in scenes:
        scene_dir = root / s
        if not scene_dir.exists():
            print(f"[skip] {scene_dir} not found")
            continue
        run_scene(model, scene_dir, out_root, device)

    print(f"[done] All 5bag adjacent-pair results (Megadepth input) written under {out_root}")

if __name__ == "__main__":
    main()
