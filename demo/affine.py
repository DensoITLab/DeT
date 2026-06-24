import cv2
import numpy as np

IMG_1_PATH = '../assets/triplet2/image1.jpg'
IMG_2_PATH = '../assets/triplet2/image2.jpg'
IMG_3_PATH = '../assets/triplet2/image3.jpg'


IMG_1_PATH = '../assets/triplet1/tri1.jpg'
IMG_2_PATH = '../assets/triplet1/tri2.jpg'
IMG_3_PATH = '../assets/triplet1/tri3.jpg'


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
IMG_11_PATH = '../assets/track_straight/Img_PVM_Front_002410.png'


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
IMG_11_PATH = '../assets/track_rotate/Img_PVM_Front_002010.png'



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
IMG_11_PATH = '../assets/track_rotate/Img_PVM_Front_002110.png'

IMG_1_PATH = '../assets/triplet2/image1.jpg'
IMG_2_PATH = '../assets/triplet2/image2.jpg'
IMG_3_PATH = '../assets/triplet2/image3.jpg'

OUTPUT_DIR = 'output_track_rotate_Test/'

VALID_IDX = 2

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
        '--image11', type=str, default=IMG_11_PATH,
        help='Path to the target image')
    parser.add_argument(
        '--output_dir', type=str, default=OUTPUT_DIR,
        help='Path of the outputs')

    opt = parser.parse_args()
    Path(opt.output_dir).mkdir(exist_ok=True, parents=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    jamma = JamMa(config=cfg, pretrained='./jamma.ckpt').eval().to(device)

    for i in range(VALID_IDX+1):
        if i == 0:
            image0, scale0, mask0, prepad_size0 = read_megadepth_color(opt.image1, 832, 16, True)
            mask0 = F.interpolate(mask0[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        elif i == 1:
            image1, scale1, mask1, prepad_size1 = read_megadepth_color(opt.image2, 832, 16, True)
            mask1 = F.interpolate(mask1[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        elif i == 2:
            image2, scale2, mask2, prepad_size2 = read_megadepth_color(opt.image3, 832, 16, True)
            mask2 = F.interpolate(mask2[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        elif i == 3:
            image3, scale3, mask3, prepad_size3 = read_megadepth_color(opt.image4, 832, 16, True)
            mask3 = F.interpolate(mask3[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        elif i == 4:
            image4, scale4, mask4, prepad_size4 = read_megadepth_color(opt.image5, 832, 16, True)
            mask4 = F.interpolate(mask4[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        elif i == 5:
            image5, scale5, mask5, prepad_size5 = read_megadepth_color(opt.image6, 832, 16, True)
            mask5 = F.interpolate(mask5[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        elif i == 6:        
            image6, scale6, mask6, prepad_size6 = read_megadepth_color(opt.image7, 832, 16, True)
            mask6 = F.interpolate(mask6[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        elif i == 7:
            image7, scale7, mask7, prepad_size7 = read_megadepth_color(opt.image8, 832, 16, True)
            mask7 = F.interpolate(mask7[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        elif i == 8:
            image8, scale8, mask8, prepad_size8 = read_megadepth_color(opt.image9, 832, 16, True)
            mask8 = F.interpolate(mask8[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        elif i == 9:
            image9, scale9, mask9, prepad_size9 = read_megadepth_color(opt.image10, 832, 16, True)
            mask9 = F.interpolate(mask9[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
        elif i == 10:
            image10, scale10, mask10, prepad_size10 = read_megadepth_color(opt.image11, 832, 16, True)
            mask10 = F.interpolate(mask10[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()         

    for i in range(VALID_IDX):
        if i == 0:
            data_12 = {
                'imagec_0': image0.to(device),
                'imagec_1': image1.to(device),
                'mask0': mask0.to(device),
                'mask1': mask1.to(device),
            }

            logger.info(f"Matching: {opt.image1} and {opt.image2}")
            jamma(data_12)
        if i == 1:
            data_23 = {
                'imagec_0': image1.to(device),
                'imagec_1': image2.to(device),
                'mask0': mask1.to(device),
                'mask1': mask2.to(device),
                'prev_data': data_12,  # Pass the previous data for consistency
            }

            logger.info(f"Matching: {opt.image2} and {opt.image3}")
            jamma(data_23)
            logger.info(f"Finish Matching, Visualizing")
            _, cor_indices = make_confidence_figure_tri(data_12, data_23, path=opt.output_dir+'viz123_res_c.png', dpi=300, topk=20, start_idx=1)
        if i == 2:
            data_34 = {
                'imagec_0': image2.to(device),
                'imagec_1': image3.to(device),
                'mask0': mask2.to(device),
                'mask1': mask3.to(device),
                'prev_data': data_23,  # Pass the previous data for consistency
            }

            logger.info(f"Matching: {opt.image3} and {opt.image4}")
            jamma(data_34)
            logger.info(f"Finish Matching, Visualizing")
            _, cor_indices = make_confidence_figure_tri(data_23, data_34, path=opt.output_dir+'viz234_res_c.png', dpi=300, topk=20, start_idx=2, cor_indices=cor_indices)
        if i == 3:
            data_45 = {
                'imagec_0': image3.to(device),
                'imagec_1': image4.to(device),
                'mask0': mask3.to(device),
                'mask1': mask4.to(device),
                'prev_data': data_34,  # Pass the previous data for consistency
            }

            logger.info(f"Matching: {opt.image4} and {opt.image5}")
            jamma(data_45)
            logger.info(f"Finish Matching, Visualizing")
            _, cor_indices = make_confidence_figure_tri(data_34, data_45, path=opt.output_dir+'viz345_res_c.png', dpi=300, topk=20, start_idx=3, cor_indices=cor_indices)
        if i == 4:
            data_56 = {
                'imagec_0': image4.to(device),
                'imagec_1': image5.to(device),
                'mask0': mask4.to(device),
                'mask1': mask5.to(device),
                'prev_data': data_45,  # Pass the previous data for consistency
            }

            logger.info(f"Matching: {opt.image5} and {opt.image6}")
            jamma(data_56)
            logger.info(f"Finish Matching, Visualizing")
            _, cor_indices = make_confidence_figure_tri(data_45, data_56, path=opt.output_dir+'viz456_res_c.png', dpi=300, topk=20, start_idx=4, cor_indices=cor_indices)
        if i == 5:
            data_67 = {
                'imagec_0': image5.to(device),
                'imagec_1': image6.to(device),
                'mask0': mask5.to(device),
                'mask1': mask6.to(device),
                'prev_data': data_56,  # Pass the previous data for consistency
            }

            logger.info(f"Matching: {opt.image6} and {opt.image7}")
            jamma(data_67)
            logger.info(f"Finish Matching, Visualizing")
            _, cor_indices = make_confidence_figure_tri(data_56, data_67, path=opt.output_dir+'viz567_res_c.png', dpi=300, topk=20, start_idx=5, cor_indices=cor_indices)
        if i == 6:
            data_78 = {
                'imagec_0': image6.to(device),
                'imagec_1': image7.to(device),
                'mask0': mask6.to(device),
                'mask1': mask7.to(device),
                'prev_data': data_67,  # Pass the previous data for consistency
            }

            logger.info(f"Matching: {opt.image7} and {opt.image8}")
            jamma(data_78)
            logger.info(f"Finish Matching, Visualizing")
            _, cor_indices = make_confidence_figure_tri(data_67, data_78, path=opt.output_dir+'viz678_res_c.png', dpi=300, topk=20, start_idx=6, cor_indices=cor_indices)
        if i == 7:
            data_89 = {
                'imagec_0': image7.to(device),
                'imagec_1': image8.to(device),
                'mask0': mask7.to(device),
                'mask1': mask8.to(device),
                'prev_data': data_78,  # Pass the previous data for consistency
            }

            logger.info(f"Matching: {opt.image8} and {opt.image9}")
            jamma(data_89)
            logger.info(f"Finish Matching, Visualizing")
            _, cor_indices = make_confidence_figure_tri(data_78, data_89, path=opt.output_dir+'viz789_res_c.png', dpi=300, topk=20, start_idx=7, cor_indices=cor_indices)
        if i == 8:
            data_910 = {
                'imagec_0': image8.to(device),
                'imagec_1': image9.to(device),
                'mask0': mask8.to(device),
                'mask1': mask9.to(device),
                'prev_data': data_89,  # Pass the previous data for consistency
            }

            logger.info(f"Matching: {opt.image9} and {opt.image10}")
            jamma(data_910)
            logger.info(f"Finish Matching, Visualizing")
            _, cor_indices = make_confidence_figure_tri(data_89, data_910, path=opt.output_dir+'viz910_res_c.png', dpi=300, topk=20, start_idx=8, cor_indices=cor_indices)
        if i == 9:
            data_1011 = {
                'imagec_0': image9.to(device),
                'imagec_1': image10.to(device),
                'mask0': mask9.to(device),
                'mask1': mask10.to(device),
                'prev_data': data_910,  # Pass the previous data for consistency
            }

            logger.info(f"Matching: {opt.image10} and {opt.image11}")
            jamma(data_1011)
            logger.info(f"Finish Matching, Visualizing")
            _, cor_indices = make_confidence_figure_tri(data_910, data_1011, path=opt.output_dir+'viz1011_res_c.png', dpi=300, topk=20, start_idx=9, cor_indices=cor_indices)


    #make_confidence_figure_tri(data_12, data_23, path=opt.output_dir+'viz2_res_c.png', dpi=900, topk=20, start_idx=0)
    #make_evaluation_figure_wheel(data, path=opt.output_dir+'viz2.png', topk=20)
    logger.info(f"Done")
