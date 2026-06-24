import os
os.sys.path.append("../")  # Add the project directory
from pathlib import Path
import torch
from utlis import JamMa, cfg
from src.utils.dataset import read_megadepth_color
import argparse
from loguru import logger
import torch.nn.functional as F
from src.utils.plotting import make_confidence_figure, make_evaluation_figure_wheel

IMG_1_PATH = '../assets/triplet2/image2.jpg'
IMG_2_PATH = '../assets/triplet2/image3.jpg'
OUTPUT_DIR = 'output_2/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching with JamMa',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--image1', type=str, default=IMG_1_PATH,
        help='Path to the source image')
    parser.add_argument(
        '--image2', type=str, default=IMG_2_PATH,
        help='Path to the target image')
    parser.add_argument(
        '--output_dir', type=str, default=OUTPUT_DIR,
        help='Path of the outputs')

    opt = parser.parse_args()
    Path(opt.output_dir).mkdir(exist_ok=True, parents=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    jamma = JamMa(config=cfg, pretrained='./jamma.ckpt').eval().to(device)

    image0, scale0, mask0, prepad_size0 = read_megadepth_color(opt.image1, 832, 16, True)
    image1, scale1, mask1, prepad_size1 = read_megadepth_color(opt.image2, 832, 16, True)
    mask0 = F.interpolate(mask0[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
    mask1 = F.interpolate(mask1[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
    data = {
        'imagec_0': image0.to(device),
        'imagec_1': image1.to(device),
        'mask0': mask0.to(device),
        'mask1': mask1.to(device),
    }

    logger.info(f"Matching: {opt.image1} and {opt.image2}")
    jamma(data)
    logger.info(f"Finish Matching, Visualizing")

    make_confidence_figure(data, path=opt.output_dir+'viz1.png', dpi=300, topk=20)
    make_evaluation_figure_wheel(data, path=opt.output_dir+'viz2.png', topk=20)
    logger.info(f"Done")
