import os
os.sys.path.append("../")  # Add the project directory
from pathlib import Path
import torch
from utlis import JamMa, cfg
from src.utils.dataset import read_megadepth_color
import argparse
from loguru import logger
import torch.nn.functional as F
from src.utils.plotting import make_confidence_figure, make_evaluation_figure_wheel, make_confidence_figure_tri, make_confidence_figure_track

from lightglue import LightGlue, SuperPoint
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

IMG_1_PATH = '../assets/track_straight/Img_PVM_Front_002400.png'
IMG_2_PATH = '../assets/track_straight/Img_PVM_Front_002401.png'
IMG_3_PATH = '../assets/track_straight/Img_PVM_Front_002402.png'
IMG_4_PATH = '../assets/track_straight/Img_PVM_Front_002403.png'
IMG_5_PATH = '../assets/track_straight/Img_PVM_Front_002404.png'
IMG_6_PATH = '../assets/track_straight/Img_PVM_Front_002405.png'
IMG_7_PATH = '../assets/track_straight/Img_PVM_Front_002406.png'
IMG_8_PATH = '../assets/track_straight/Img_PVM_Front_002407.png'
IMG_9_PATH = '../assets/track_straight/Img_PVM_Front_002408.png'
IMG_10_PATH = '../assets/track_straight/Img_PVM_Front_002409.png'



"""
IMG_1_PATH = '../assets/track_straight_fcm/Img_FCM_FCM1_002400.png'
IMG_2_PATH = '../assets/track_straight_fcm/Img_FCM_FCM1_002401.png'
IMG_3_PATH = '../assets/track_straight_fcm/Img_FCM_FCM1_002402.png'
IMG_4_PATH = '../assets/track_straight_fcm/Img_FCM_FCM1_002403.png'
IMG_5_PATH = '../assets/track_straight_fcm/Img_FCM_FCM1_002404.png'
IMG_6_PATH = '../assets/track_straight_fcm/Img_FCM_FCM1_002405.png'
IMG_7_PATH = '../assets/track_straight_fcm/Img_FCM_FCM1_002406.png'
IMG_8_PATH = '../assets/track_straight_fcm/Img_FCM_FCM1_002407.png'
IMG_9_PATH = '../assets/track_straight_fcm/Img_FCM_FCM1_002408.png'
IMG_10_PATH = '../assets/track_straight_fcm/Img_FCM_FCM1_002409.png'

IMG_1_PATH = '../assets/track_rotate/Img_PVM_Front_002000.png' 
IMG_2_PATH = '../assets/track_rotate/Img_PVM_Front_002001.png'
IMG_3_PATH = '../assets/track_rotate/Img_PVM_Front_002002.png'
IMG_4_PATH = '../assets/track_rotate/Img_PVM_Front_002003.png'
IMG_5_PATH = '../assets/track_rotate/Img_PVM_Front_002004.png'
IMG_6_PATH = '../assets/track_rotate/Img_PVM_Front_002005.png'
IMG_7_PATH = '../assets/track_rotate/Img_PVM_Front_002006.png'
IMG_8_PATH = '../assets/track_rotate/Img_PVM_Front_002007.png'
IMG_9_PATH = '../assets/track_rotate/Img_PVM_Front_002008.png'
IMG_10_PATH = '../assets/track_rotate/Img_PVM_Front_002009.png'



IMG_1_PATH = '../assets/track_rotate/Img_PVM_Front_002010.png' 
IMG_2_PATH = '../assets/track_rotate/Img_PVM_Front_002020.png'
IMG_3_PATH = '../assets/track_rotate/Img_PVM_Front_002030.png'
IMG_4_PATH = '../assets/track_rotate/Img_PVM_Front_002040.png'
IMG_5_PATH = '../assets/track_rotate/Img_PVM_Front_002050.png'
IMG_6_PATH = '../assets/track_rotate/Img_PVM_Front_002060.png'
IMG_7_PATH = '../assets/track_rotate/Img_PVM_Front_002070.png'
IMG_8_PATH = '../assets/track_rotate/Img_PVM_Front_002080.png'
IMG_9_PATH = '../assets/track_rotate/Img_PVM_Front_002090.png'
IMG_10_PATH = '../assets/track_rotate/Img_PVM_Front_002100.png'
"""
OUTPUT_DIR = 'output_track_straight/'
OUTPUT_DIR2 = 'output_matches_straight/'


VALID_IDX = 2

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
    parser.add_argument(
        '--output_dir2', type=str, default=OUTPUT_DIR2,
        help='Path of the outputs')

    opt = parser.parse_args()
    Path(opt.output_dir).mkdir(exist_ok=True, parents=True)
    Path(opt.output_dir2).mkdir(exist_ok=True, parents=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg['is_finelevel'] = False
    jamma = JamMa(config=cfg, pretrained='./jamma.ckpt').eval().to(device)
    extractor = SuperPoint(max_num_keypoints=None, detection_threshold=0.00001).eval().to(device)  # load the extractor


    
    image0, scale0, mask0, prepad_size0, origin_wh, new_wh = read_megadepth_color(opt.image1, 832, 16, True)
    mask0 = F.interpolate(mask0[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
    
    image1, scale1, mask1, prepad_size1, _,_ = read_megadepth_color(opt.image2, 832, 16, True)
    mask1 = F.interpolate(mask1[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        
    image2, scale2, mask2, prepad_size2,_,_ = read_megadepth_color(opt.image3, 832, 16, True)
    mask2 = F.interpolate(mask2[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        
    image3, scale3, mask3, prepad_size3,_,_ = read_megadepth_color(opt.image4, 832, 16, True)
    mask3 = F.interpolate(mask3[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
       
    image4, scale4, mask4, prepad_size4,_,_ = read_megadepth_color(opt.image5, 832, 16, True)
    mask4 = F.interpolate(mask4[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        
    image5, scale5, mask5, prepad_size5,_,_ = read_megadepth_color(opt.image6, 832, 16, True)
    mask5 = F.interpolate(mask5[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
               
    image6, scale6, mask6, prepad_size6,_,_ = read_megadepth_color(opt.image7, 832, 16, True)
    mask6 = F.interpolate(mask6[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        
    image7, scale7, mask7, prepad_size7,_,_ = read_megadepth_color(opt.image8, 832, 16, True)
    mask7 = F.interpolate(mask7[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        
    image8, scale8, mask8, prepad_size8,_,_ = read_megadepth_color(opt.image9, 832, 16, True)
    mask8 = F.interpolate(mask8[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        
    image9, scale9, mask9, prepad_size9,_,_ = read_megadepth_color(opt.image10, 832, 16, True)
    mask9 = F.interpolate(mask9[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()

    image0_sp = extractor.extract(image0[0].to(device))
    image1_sp = extractor.extract(image1[0].to(device))
    image2_sp = extractor.extract(image2[0].to(device))
    image3_sp = extractor.extract(image3[0].to(device))
    image4_sp = extractor.extract(image4[0].to(device))
    image5_sp = extractor.extract(image5[0].to(device))
    image6_sp = extractor.extract(image6[0].to(device))
    image7_sp = extractor.extract(image7[0].to(device))
    image8_sp = extractor.extract(image8[0].to(device))
    image9_sp = extractor.extract(image9[0].to(device))

    image0_sp = image0_sp['keypoints'][0]
    image1_sp = image1_sp['keypoints'][0]
    image2_sp = image2_sp['keypoints'][0]
    image3_sp = image3_sp['keypoints'][0]
    image4_sp = image4_sp['keypoints'][0]
    image5_sp = image5_sp['keypoints'][0]
    image6_sp = image6_sp['keypoints'][0]
    image7_sp = image7_sp['keypoints'][0]
    image8_sp = image8_sp['keypoints'][0]
    image9_sp = image9_sp['keypoints'][0]
    print(image0_sp.shape, image1_sp.shape)
    print(image2_sp.shape, image3_sp.shape)
    print(image4_sp.shape, image5_sp.shape)
    print(image6_sp.shape, image7_sp.shape)
    print(image8_sp.shape, image9_sp.shape)

    data_12 = {
                'imagec_0': image0.to(device),
                'imagec_1': image1.to(device),
                'mask0': mask0.to(device),
                'mask1': mask1.to(device),
                'algo_res' : ALGO_RES,
                'sp0': image0_sp,
                'sp1': image1_sp
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
                'sp0': image1_sp,
                'sp1': image2_sp
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
                'sp0': image2_sp,
                'sp1': image3_sp
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
                'sp0': image3_sp,
                'sp1': image4_sp
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
                'sp0': image4_sp,
                'sp1': image5_sp
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
                'sp0': image5_sp,
                'sp1': image6_sp
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
                'sp0': image6_sp,
                'sp1': image7_sp
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
                'sp0': image7_sp,
                'sp1': image8_sp
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
                'sp0': image8_sp,
                'sp1': image9_sp
            }
        logger.info(f"Matching: {opt.image9} and {opt.image10}")
        jamma(data_910)
    logger.info(f"Finish Matching, Visualizing")
    
    result_path = opt.output_dir2+'res5000conf_res_ratest_conf_sp_wide.json'
    #result_path = None
    make_confidence_figure_track(data_12, data_23, data_34, data_45, data_56, data_67, data_78, data_89, data_910,
                                 path=opt.output_dir+'track_result_res5000conf_res_sp_all_wide.jpg', dpi=700, topk=5000, mode=2, _all=False, add_mode=2,
                                 result_path=result_path, is_origin_img=False, origin_wh=origin_wh, new_wh=new_wh, draw_line=True, img_num=VALID_IDX+1, is_roi=True, super_point=True)
    #make_evaluation_figure_wheel(data, path=opt.output_dir+'viz2.png', topk=20)
    logger.info(f"Done")
