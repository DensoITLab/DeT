import os
os.sys.path.append("../")  # Add the project directory
from pathlib import Path
import torch
from utlis import JamMa, cfg
from src.utils.dataset import read_megadepth_color
import argparse
from loguru import logger
import torch.nn.functional as F
from src.utils.plotting import make_confidence_figure, make_evaluation_figure_wheel, make_confidence_figure_tri, make_confidence_figure_track, make_confidence_figure_track_compare

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import time
import numpy as np
from thop import profile

# 画像パス設定（IMG1~IMG10まで）
IMG_1_PATH = '/home/ach17765lb/JamMa/demo/guernica_seq_fixed/img0.png'
IMG_2_PATH = '/home/ach17765lb/JamMa/demo/guernica_seq_fixed/img1.png'
IMG_3_PATH = '/home/ach17765lb/JamMa/demo/guernica_seq_fixed/img2.png'
IMG_4_PATH = '/home/ach17765lb/JamMa/demo/guernica_seq_fixed/img3.png'
IMG_5_PATH = '/home/ach17765lb/JamMa/demo/guernica_seq_fixed/img4.png'
IMG_6_PATH = '/home/ach17765lb/JamMa/demo/guernica_seq_fixed/img5.png'
IMG_7_PATH = '/home/ach17765lb/JamMa/demo/guernica_seq_fixed/img6.png'
IMG_8_PATH = '/home/ach17765lb/JamMa/demo/guernica_seq_fixed/img7.png'
IMG_9_PATH = '/home/ach17765lb/JamMa/demo/guernica_seq_fixed/img8.png'
IMG_10_PATH = '/home/ach17765lb/JamMa/demo/guernica_seq_fixed/img9.png'


#IMG_1_PATH = '/home/ach17765lb/JamMa/assets/carla1/frame_00002.png'
#IMG_2_PATH = '/home/ach17765lb/JamMa/assets/carla1/frame_00003.png'
#IMG_3_PATH = '/home/ach17765lb/JamMa/assets/carla1/frame_00004.png'
#IMG_4_PATH = '/home/ach17765lb/JamMa/assets/carla1/frame_00005.png'
#IMG_5_PATH = '/home/ach17765lb/JamMa/assets/carla1/frame_00006.png'
#IMG_6_PATH = '/home/ach17765lb/JamMa/assets/carla1/frame_00007.png'
#IMG_7_PATH = '/home/ach17765lb/JamMa/assets/carla1/frame_00008.png'
#IMG_8_PATH = '/home/ach17765lb/JamMa/assets/carla1/frame_00009.png'
#IMG_9_PATH = '/home/ach17765lb/JamMa/assets/carla1/frame_00010.png'
#IMG_10_PATH = '/home/ach17765lb/JamMa/assets/carla1/frame_00011.png'

"""
IMG_1_PATH = '/home/ach17765lb/JamMa/assets/carla2/frame_00191.png'
IMG_2_PATH = '/home/ach17765lb/JamMa/assets/carla2/frame_00192.png'
IMG_3_PATH = '/home/ach17765lb/JamMa/assets/carla2/frame_00193.png'
IMG_4_PATH = '/home/ach17765lb/JamMa/assets/carla2/frame_00194.png'
IMG_5_PATH = '/home/ach17765lb/JamMa/assets/carla2/frame_00195.png'
IMG_6_PATH = '/home/ach17765lb/JamMa/assets/carla2/frame_00196.png'
IMG_7_PATH = '/home/ach17765lb/JamMa/assets/carla2/frame_00197.png'
IMG_8_PATH = '/home/ach17765lb/JamMa/assets/carla2/frame_00198.png'
IMG_9_PATH = '/home/ach17765lb/JamMa/assets/carla2/frame_00199.png'
IMG_10_PATH = '/home/ach17765lb/JamMa/assets/carla2/frame_00200.png'

IMG_1_PATH = '/home/ach17765lb/JamMa/assets/carla3/frame_00035.png'
IMG_2_PATH = '/home/ach17765lb/JamMa/assets/carla3/frame_00036.png'
IMG_3_PATH = '/home/ach17765lb/JamMa/assets/carla3/frame_00037.png'
IMG_4_PATH = '/home/ach17765lb/JamMa/assets/carla3/frame_00038.png'
IMG_5_PATH = '/home/ach17765lb/JamMa/assets/carla3/frame_00039.png'
IMG_6_PATH = '/home/ach17765lb/JamMa/assets/carla3/frame_00040.png'
IMG_7_PATH = '/home/ach17765lb/JamMa/assets/carla3/frame_00041.png'
IMG_8_PATH = '/home/ach17765lb/JamMa/assets/carla3/frame_00042.png'
IMG_9_PATH = '/home/ach17765lb/JamMa/assets/carla3/frame_00043.png'
IMG_10_PATH = '/home/ach17765lb/JamMa/assets/carla3/frame_00044.png'

IMG_1_PATH = '/home/ach17765lb/JamMa/assets/carla4/frame_00180.png'
IMG_2_PATH = '/home/ach17765lb/JamMa/assets/carla4/frame_00181.png'
IMG_3_PATH = '/home/ach17765lb/JamMa/assets/carla4/frame_00182.png'
IMG_4_PATH = '/home/ach17765lb/JamMa/assets/carla4/frame_00183.png'
IMG_5_PATH = '/home/ach17765lb/JamMa/assets/carla4/frame_00184.png'
IMG_6_PATH = '/home/ach17765lb/JamMa/assets/carla4/frame_00185.png'
IMG_7_PATH = '/home/ach17765lb/JamMa/assets/carla4/frame_00186.png'
IMG_8_PATH = '/home/ach17765lb/JamMa/assets/carla4/frame_00187.png'
IMG_9_PATH = '/home/ach17765lb/JamMa/assets/carla4/frame_00188.png'
IMG_10_PATH = '/home/ach17765lb/JamMa/assets/carla4/frame_00189.png'

IMG_1_PATH = '/home/ach17765lb/JamMa/assets/carla5/frame_00500.png'
IMG_2_PATH = '/home/ach17765lb/JamMa/assets/carla5/frame_00501.png'
IMG_3_PATH = '/home/ach17765lb/JamMa/assets/carla5/frame_00502.png'
IMG_4_PATH = '/home/ach17765lb/JamMa/assets/carla5/frame_00503.png'
IMG_5_PATH = '/home/ach17765lb/JamMa/assets/carla5/frame_00504.png'
IMG_6_PATH = '/home/ach17765lb/JamMa/assets/carla5/frame_00505.png'
IMG_7_PATH = '/home/ach17765lb/JamMa/assets/carla5/frame_00506.png'
IMG_8_PATH = '/home/ach17765lb/JamMa/assets/carla5/frame_00507.png'
IMG_9_PATH = '/home/ach17765lb/JamMa/assets/carla5/frame_00508.png'
IMG_10_PATH = '/home/ach17765lb/JamMa/assets/carla5/frame_00509.png'
"""

IMG_1_PATH = "/home/ach17765lb/JamMa/assets/test_subset10/img0.jpg"
IMG_2_PATH = "/home/ach17765lb/JamMa/assets/test_subset10/img1.jpg"
IMG_3_PATH = "/home/ach17765lb/JamMa/assets/test_subset10/img2.jpg"
IMG_4_PATH = "/home/ach17765lb/JamMa/assets/test_subset10/img3.jpg"
IMG_5_PATH = "/home/ach17765lb/JamMa/assets/test_subset10/img4.jpg"
IMG_6_PATH = "/home/ach17765lb/JamMa/assets/test_subset10/img5.jpg"
IMG_7_PATH = "/home/ach17765lb/JamMa/assets/test_subset10/img6.jpg"
IMG_8_PATH = "/home/ach17765lb/JamMa/assets/test_subset10/img7.jpg"
IMG_9_PATH = "/home/ach17765lb/JamMa/assets/test_subset10/img8.jpg"
IMG_10_PATH = "/home/ach17765lb/JamMa/assets/test_subset10/img9.jpg"


OUTPUT_DIR = 'output_guernica_seq_fixed/'
OUTPUT_DIR = 'output_test_subset10/'

#OUTPUT_DIR = 'output_carla1/'


VALID_IDX = 9 # 有効画像数（1始まり）。9=> img0~img9までの10枚処理

ALGO_RES = True

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
        '--image3', type=str, default=IMG_3_PATH,
        help='Path to the target image')
    parser.add_argument(
        '--image4', type=str, default=IMG_4_PATH,
        help='Path to the target image')
    parser.add_argument(
        '--image5', type=str, default=IMG_5_PATH,
        help='Path to the target image')
    parser.add_argument(
        '--image6', type=str, default=IMG_6_PATH,
        help='Path to the target image')
    parser.add_argument(
        '--image7', type=str, default=IMG_7_PATH,
        help='Path to the target image')
    parser.add_argument(
        '--image8', type=str, default=IMG_8_PATH,
        help='Path to the target image')
    parser.add_argument(
        '--image9', type=str, default=IMG_9_PATH,
        help='Path to the target image')
    parser.add_argument(
        '--image10', type=str, default=IMG_10_PATH,
        help='Path to the target image')
    parser.add_argument(
        '--output_dir', type=str, default=OUTPUT_DIR,
        help='Path of the outputs')

    opt = parser.parse_args()
    Path('/home/ach17765lb/JamMa/demo/' + opt.output_dir).mkdir(exist_ok=True, parents=True)

    result_path = '/home/ach17765lb/JamMa/demo/' + opt.output_dir+'track_result.json' # 出力先 JSON パス

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg['debug'] = True
    cfg['use_flex'] = True # 提案アルゴを有効化
    shifted = False
    print(cfg, device)

    jamma = JamMa(config=cfg, pretrained='/home/ach17765lb/JamMa/demo/jamma.ckpt').eval().to(device)

    image0, scale0, mask0, prepad_size0, origin_wh0, new_wh0 = read_megadepth_color(opt.image1, 832, 8, True)
    mask0 = F.interpolate(mask0[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
    
    image1, scale1, mask1, prepad_size1, origin_wh1, new_wh1 = read_megadepth_color(opt.image2, 832, 8, True)
    mask1 = F.interpolate(mask1[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        
    image2, scale2, mask2, prepad_size2, origin_wh2, new_wh2 = read_megadepth_color(opt.image3, 832, 8, True)
    mask2 = F.interpolate(mask2[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        
    image3, scale3, mask3, prepad_size3, origin_wh3, new_wh3 = read_megadepth_color(opt.image4, 832, 8, True)
    mask3 = F.interpolate(mask3[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
       
    image4, scale4, mask4, prepad_size4, origin_wh4, new_wh4 = read_megadepth_color(opt.image5, 832, 8, True)
    mask4 = F.interpolate(mask4[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        
    image5, scale5, mask5, prepad_size5, origin_wh5, new_wh5 = read_megadepth_color(opt.image6, 832, 8, True)
    mask5 = F.interpolate(mask5[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
               
    image6, scale6, mask6, prepad_size6, origin_wh6, new_wh6 = read_megadepth_color(opt.image7, 832, 8, True)
    mask6 = F.interpolate(mask6[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        
    image7, scale7, mask7, prepad_size7, origin_wh7, new_wh7 = read_megadepth_color(opt.image8, 832, 8, True)
    mask7 = F.interpolate(mask7[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        
    image8, scale8, mask8, prepad_size8, origin_wh8, new_wh8 = read_megadepth_color(opt.image9, 832, 8, True)
    mask8 = F.interpolate(mask8[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        
    image9, scale9, mask9, prepad_size9, origin_wh9, new_wh9 = read_megadepth_color(opt.image10, 832, 8, True)
    mask9 = F.interpolate(mask9[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()

    dx, dy=0, 0
    if shifted:
        dx, dy = 6,6
        # padは(left, right, top, bottom) の順
        padded  = F.pad(image0, (dx, 0, dy, 0), mode='constant', value=0)
        image1 = padded[:, :, :image0.shape[2], :image0.shape[3]]   # 元サイズにトリミング

        padded  = F.pad(image0, (dx*2, 0, dy*2, 0), mode='constant', value=0)
        image2 = padded[:, :, :image0.shape[2], :image0.shape[3]]   # 元サイズにトリミング

        padded  = F.pad(image0, (dx*3, 0, dy*3, 0), mode='constant', value=0)
        image3 = padded[:, :, :image0.shape[2], :image0.shape[3]]   # 元サイズにトリミング

        padded  = F.pad(image0, (dx*4, 0, dy*4, 0), mode='constant', value=0)
        image4 = padded[:, :, :image0.shape[2], :image0.shape[3]]   # 元サイズにトリミング

        padded  = F.pad(image0, (dx*5, 0, dy*5, 0), mode='constant', value=0)
        image5 = padded[:, :, :image0.shape[2], :image0.shape[3]]   # 元サイズにトリミング

        padded  = F.pad(image0, (dx*6, 0, dy*6, 0), mode='constant', value=0)
        image6 = padded[:, :, :image0.shape[2], :image0.shape[3]]   # 元サイズにトリミング

        padded  = F.pad(image0, (dx*7, 0, dy*7, 0), mode='constant', value=0)
        image7 = padded[:, :, :image0.shape[2], :image0.shape[3]]   # 元サイズにトリミング

        padded  = F.pad(image0, (dx*8, 0, dy*8, 0), mode='constant', value=0)
        image8 = padded[:, :, :image0.shape[2], :image0.shape[3]]   # 元サイズにトリミング
        
        padded  = F.pad(image0, (dx*9, 0, dy*9, 0), mode='constant', value=0)
        image9 = padded[:, :, :image0.shape[2], :image0.shape[3]]   # 元サイズにトリミング

    #H01 = np.load('/home/ach17765lb/JamMa/demo/guernica_seq_fixed/H_01.npy')
    #H12 = np.load('/home/ach17765lb/JamMa/demo/guernica_seq_fixed/H_12.npy')
    #H23 = np.load('/home/ach17765lb/JamMa/demo/guernica_seq_fixed/H_23.npy')
    #H34 = np.load('/home/ach17765lb/JamMa/demo/guernica_seq_fixed/H_34.npy')
    #H45 = np.load('/home/ach17765lb/JamMa/demo/guernica_seq_fixed/H_45.npy')
    #H56 = np.load('/home/ach17765lb/JamMa/demo/guernica_seq_fixed/H_56.npy')
    #H67 = np.load('/home/ach17765lb/JamMa/demo/guernica_seq_fixed/H_67.npy')
    #H78 = np.load('/home/ach17765lb/JamMa/demo/guernica_seq_fixed/H_78.npy')
    #H89 = np.load('/home/ach17765lb/JamMa/demo/guernica_seq_fixed/H_89.npy')

    data_12 = {
                'imagec_0': image0.to(device),
                'imagec_1': image1.to(device),
                'mask0': mask0.to(device),
                'mask1': mask1.to(device),
                'algo_res' : ALGO_RES,
                #'H_shift': H01
            }
    logger.info(f"Matching: {opt.image1} and {opt.image2}")
    jamma(data_12)

    data_23 = None
    data_34 = None
    data_45 = None
    data_56 = None
    data_67 = None
    data_78 = None
    data_89 = None
    data_910 = None

    if VALID_IDX >=2:
        data_23 = {
                'imagec_0': image1.to(device),
                'imagec_1': image2.to(device),
                'mask0': mask1.to(device),
                'mask1': mask2.to(device),
                'prev_data': data_12,  # Pass the previous data for consistency
                'algo_res' : ALGO_RES,
                #'H_shift': H12
                }

        logger.info(f"Matching: {opt.image2} and {opt.image3}")
        jamma(data_23)


    if VALID_IDX >=3:
        data_34 = {
                'imagec_0': image2.to(device),
                'imagec_1': image3.to(device),
                'mask0': mask2.to(device),
                'mask1': mask3.to(device),
                'prev_data': data_23,  # Pass the previous data for consistency
                'algo_res' : ALGO_RES,
                #'H_shift': H23
            }

        logger.info(f"Matching: {opt.image3} and {opt.image4}")
        jamma(data_34)
        

    if VALID_IDX >=4:
        data_45 = {
                'imagec_0': image3.to(device),
                'imagec_1': image4.to(device),
                'mask0': mask3.to(device),
                'mask1': mask4.to(device),
                'prev_data': data_34,  # Pass the previous data for consistency
                'algo_res' : ALGO_RES,
                #'H_shift': H34
            }

        logger.info(f"Matching: {opt.image4} and {opt.image5}")
        jamma(data_45)


    if VALID_IDX >=5:
        data_56 = {
                'imagec_0': image4.to(device),
                'imagec_1': image5.to(device),
                'mask0': mask4.to(device),
                'mask1': mask5.to(device),
                'prev_data': data_45,  # Pass the previous data for consistency
                'algo_res' : ALGO_RES,
                #'H_shift': H45
            }

        logger.info(f"Matching: {opt.image5} and {opt.image6}")
        jamma(data_56)


    if VALID_IDX >=6:
        data_67 = {
                'imagec_0': image5.to(device),
                'imagec_1': image6.to(device),
                'mask0': mask5.to(device),
                'mask1': mask6.to(device),
                'prev_data': data_56,  # Pass the previous data for consistency
                'algo_res' : ALGO_RES,
                #'H_shift': H56
            }

        logger.info(f"Matching: {opt.image6} and {opt.image7}")
        jamma(data_67)


    if VALID_IDX >=7:
        data_78 = {
                'imagec_0': image6.to(device),
                'imagec_1': image7.to(device),
                'mask0': mask6.to(device),
                'mask1': mask7.to(device),
                'prev_data': data_67,  # Pass the previous data for consistency
                'algo_res' : ALGO_RES,
                #'H_shift': H67
            }
        logger.info(f"Matching: {opt.image7} and {opt.image8}")
        jamma(data_78)


    if VALID_IDX >=8:
        data_89 = {
                'imagec_0': image7.to(device),
                'imagec_1': image8.to(device),
                'mask0': mask7.to(device),
                'mask1': mask8.to(device),
                'prev_data': data_78,  # Pass the previous data for consistency
                'algo_res' : ALGO_RES,
                #'H_shift': H78
            }
        logger.info(f"Matching: {opt.image8} and {opt.image9}")
        jamma(data_89)

    if VALID_IDX >=9:
        data_910 = {
                'imagec_0': image8.to(device),
                'imagec_1': image9.to(device),
                'mask0': mask8.to(device),
                'mask1': mask9.to(device),
                'prev_data': data_89,  # Pass the previous data for consistency
                'algo_res' : ALGO_RES,
                #'H_shift': H89
            }
        logger.info(f"Matching: {opt.image9} and {opt.image10}")
        jamma(data_910)

    logger.info(f"Finish Matching, Visualizing")

    make_confidence_figure_track(data_12, data_23, data_34, data_45, data_56, data_67, data_78, data_89, data_910,
                                 dpi=700, topk=15000, mode=2, _all=True, add_mode=0,
                                 result_path=result_path, is_origin_img=True, origin_wh=[origin_wh0, origin_wh1, origin_wh2, origin_wh3, origin_wh4, origin_wh5, origin_wh6, origin_wh7, origin_wh8, origin_wh9], new_wh=[new_wh0, new_wh1, new_wh2, new_wh3, new_wh4, new_wh5, new_wh6, new_wh7, new_wh8, new_wh9], draw_line=True, img_num=VALID_IDX+1, is_roi=False, dx=dx, dy=dy)

    del data_12
    del data_23
    del data_34
    del data_45
    del data_56
    del data_67
    del data_78
    del data_89
    del data_910
    
    is_compare = False
    if is_compare:
        data_01 = {
                'imagec_0': image0.to(device),
                'imagec_1': image1.to(device),
                'mask0': mask0.to(device),
                'mask1': mask1.to(device),
                'algo_res' : ALGO_RES,
                #'H_shift': H01
            }
        jamma(data_01)

        data_02 = None
        data_03 = None
        data_04 = None
        data_05 = None
        data_06 = None
        data_07 = None
        data_08 = None
        data_09 = None

        if VALID_IDX >=2:
            data_02 = {
                'imagec_0': image0.to(device),
                'imagec_1': image2.to(device),
                'mask0': mask0.to(device),
                'mask1': mask2.to(device),
                'algo_res' : ALGO_RES,
                #'H_shift': H12
                }
            jamma(data_02)


        if VALID_IDX >=3:
            data_03 = {
                'imagec_0': image0.to(device),
                'imagec_1': image3.to(device),
                'mask0': mask0.to(device),
                'mask1': mask3.to(device),
                'algo_res' : ALGO_RES,
                #'H_shift': H23
            }
            jamma(data_03)


        if VALID_IDX >=4:
            data_04 = {
                'imagec_0': image0.to(device),
                'imagec_1': image4.to(device),
                'mask0': mask0.to(device),
                'mask1': mask4.to(device),
                'algo_res' : ALGO_RES,
                #'H_shift': H34
            }
            jamma(data_04)


        if VALID_IDX >=5:
            data_05 = {
                'imagec_0': image0.to(device),
                'imagec_1': image5.to(device),
                'mask0': mask0.to(device),
                'mask1': mask5.to(device),
                'algo_res' : ALGO_RES,
                #'H_shift': H45
            }
            jamma(data_05)


        if VALID_IDX >=6:
            data_06 = {
                'imagec_0': image0.to(device),
                'imagec_1': image6.to(device),
                'mask0': mask0.to(device),
                'mask1': mask6.to(device),
                'algo_res' : ALGO_RES,
                #'H_shift': H56
            }
            jamma(data_06)


        if VALID_IDX >=7:
            data_07 = {
                'imagec_0': image0.to(device),
                'imagec_1': image7.to(device),
                'mask0': mask0.to(device),
                'mask1': mask7.to(device),
                'algo_res' : ALGO_RES,
                #'H_shift': H67
            }
            jamma(data_07)


        if VALID_IDX >=8:
            data_08 = {
                'imagec_0': image0.to(device),
                'imagec_1': image8.to(device),
                'mask0': mask0.to(device),
                'mask1': mask8.to(device),
                'prev_data': data_07,  # Pass the previous data for consistency
                'algo_res' : ALGO_RES,
                #'H_shift': H78
            }
            jamma(data_08)

        if VALID_IDX >=9:
            data_09 = {
                'imagec_0': image0.to(device),
                'imagec_1': image9.to(device),
                'mask0': mask0.to(device),
                'mask1': mask9.to(device),
                'algo_res' : ALGO_RES,
                #'H_shift': H89
            }
            jamma(data_09)

        result_path = '/home/ach17765lb/JamMa/demo/' + opt.output_dir+'track_result_compare.json' # 出力先 JSON パス
        make_confidence_figure_track_compare(data_01, data_02, data_03, data_04, data_05, data_06, data_07, data_08, data_09,
                                 topk=15000, result_path=result_path, is_origin_img=True, origin_wh=origin_wh, new_wh=new_wh)

    logger.info(f"Done")
