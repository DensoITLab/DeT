import os
os.sys.path.append("../")  # Add the project directory
from pathlib import Path
import glob
import torch
from utlis import JamMa, cfg
from src.utils.dataset import read_megadepth_color
import argparse
from loguru import logger
import torch.nn.functional as F
from src.utils.plotting import make_confidence_figure, make_evaluation_figure_wheel, make_confidence_figure_tri, make_confidence_figure_tri_res

IMG_1_PATH = '../assets/triplet2/image1.jpg'
IMG_2_PATH = '../assets/triplet2/image2.jpg'
IMG_3_PATH = '../assets/triplet2/image3.jpg'

IMG_1_PATH = '../assets/triplet1/tri1.jpg'
IMG_2_PATH = '../assets/triplet1/tri2.jpg'
IMG_3_PATH = '../assets/triplet1/tri3.jpg'

OUTPUT_DIR = 'output_tri_res1/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching with JamMa',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--image_dir', type=str, default='../assets/tripret1/',
        help='Path to the source imagedir')
    parser.add_argument(
        '--output_dir', type=str, default=OUTPUT_DIR,
        help='Path of the outputs')

    opt = parser.parse_args()
    Path(opt.output_dir).mkdir(exist_ok=True, parents=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    jamma = JamMa(config=cfg, pretrained='./jamma.ckpt').eval().to(device)
    image_paths = sorted(glob.glob(opt.image_dir + '*.jpg'))
    num_images = len(image_paths)

    for i in range(num_images -1):
        image0, scale0, mask0, prepad_size0 = read_megadepth_color(image_paths[i], 832, 16, True)
        image1, scale1, mask1, prepad_size1 = read_megadepth_color(image_paths[i + 1], 832, 16, True)

        mask0 = F.interpolate(mask0[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        mask1 = F.interpolate(mask1[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()

        print(f"Image 0: {image_paths[i]}, Image 1: {image_paths[i + 1]}")
        logger.info(f"image0 shape: {image0.shape}, image1 shape: {image1.shape}")
        logger.info(f"mask0 shape: {mask0.shape}, mask1 shape: {mask1.shape}")

        data = {
            'imagec_0': image0.to(device),
            'imagec_1': image1.to(device),
            'mask0': mask0.to(device),
            'mask1': mask1.to(device),
        }

        logger.info(f"Matching: {image_paths[i]} and {image_paths[i + 1]}")
        jamma(data)
        logger.info(f"Finish Matching, Visualizing")


    make_confidence_figure_tri_res(data, path=opt.output_dir+'viz1_res.png', dpi=900, topk=50)
    #make_evaluation_figure_wheel(data, path=opt.output_dir+'viz2.png', topk=20)
    logger.info(f"Done")
