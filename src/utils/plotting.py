import bisect
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import ConnectionPatch
import cv2
import torch
from matplotlib import cm
import matplotlib.patheffects as path_effects
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])


def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth':
        thr = 1e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr

def _get_color(flag):
    if flag == 'new':
        return 'green'
    elif flag:
        return 'cyan'
    else:
        return 'yellow'
    
def _get_color2(valid_list, end_list=[]):
    color_list = []
    for i, flag in enumerate(valid_list):
        if i in end_list:
            color_list.append('red')
        else:
            color_list.append('blue')
    return color_list

def _get_marker(valid_list, mode=0, end_list=[]):
    marker_list = []
    if mode == 0:
        for i, flag in enumerate(valid_list):
            if flag == 'new':
                marker_list.append('*')
            else:
                marker_list.append('o')
    else:
        for i, flag in enumerate(valid_list):
            if i in end_list:
                marker_list.append('x')
            else:
                marker_list.append('.')
    return marker_list


# --- VISUALIZATION --- #
def make_matching_figure_color(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(5, 3), dpi=dpi)
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=1)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=1)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=0.5)
                                        for i in range(len(mkpts0))]
        
        sc0 = axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=2)
        sc1 = axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=2)

        labels0 = [str(num) for num in range(1,len(mkpts0)+1)]
        labels1 = [str(num) for num in range(1,len(mkpts1)+1)]

        for i, label0 in enumerate(labels0):
            axes[0].text(mkpts0[:, 0][i]-10, mkpts0[:, 1][i]-10, label0, fontsize=4)

        for i, label1 in enumerate(labels1):
            axes[1].text(mkpts1[:, 0][i]-10, mkpts1[:, 1][i]-10, label1, fontsize=4)

    # put txts
    txt_color = 'k'
    text_ = fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=5, va='top', ha='left', color=txt_color)
    text_.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    # save or return figure
    if path:
        cax = fig.add_axes((0.05, 0.05, 0.9, 0.08))
        plt.colorbar(sc1, orientation='horizontal', cax=cax)
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()
    else:
        return fig



def make_matching_figure_color_tri(
        img0, img1, img2, mkpts0, mkpts1, _mkpts1, mkpts2,bind_list,valid_list,
        kpts0=None, kpts1=None, kpts2=None, text=[], dpi=75, path=None, start_idx=0):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 3, figsize=(15, 9), dpi=dpi)
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    axes[2].imshow(img2)
    for i in range(3):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        assert kpts2 is not None

        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=1)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=1)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        _fkpts1 = transFigure.transform(axes[1].transData.transform(_mkpts1))

        fkpts2 = transFigure.transform(axes[2].transData.transform(mkpts2))
        lines01 = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, linewidth=0.6)
                                        for i in range(len(mkpts0))]
        lines12 = [matplotlib.lines.Line2D((_fkpts1[i, 0], fkpts2[i, 0]),
                                            (_fkpts1[i, 1], fkpts2[i, 1]),
                                            transform=fig.transFigure, c=[_get_color(flag) for flag in valid_list][i], linewidth=0.8)
                                        for i in range(len(_mkpts1))]
        
        dist_lines = [matplotlib.lines.Line2D((_fkpts1[i, 0], fkpts1[t, 0]),
                                            (_fkpts1[i, 1], fkpts1[t, 1]),
                                            transform=fig.transFigure, c= 'green', linewidth=0.4)
                                        for i, t in bind_list]
        fig.lines.extend(lines12)
        fig.lines.extend(lines01)
        fig.lines.extend(dist_lines)
        sc0 = axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1],  s=4, alpha=1.0)
        _sc1 = axes[1].scatter(_mkpts1[:, 0], _mkpts1[:, 1],  c=[_get_color(flag) for flag in valid_list], s=10, alpha=0.5)
        sc1 = axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1],  s=4, alpha=1.0)
        sc2 = axes[2].scatter(mkpts2[:, 0], mkpts2[:, 1],  s=4, c= [_get_color(flag) for flag in valid_list], alpha=1.0)

        labels0 = [str(num) for num in range(1,len(mkpts0)+1)]
        labels1 = [str(num) for num in range(1,len(mkpts1)+1)]
        _labels1 = [str(num) for num in range(1,len(_mkpts1)+1)]
        labels2 = [str(num) for num in range(1,len(mkpts2)+1)]

        for i, label0 in enumerate(labels0):
            axes[0].text(mkpts0[:, 0][i]-10, mkpts0[:, 1][i]-10, label0, fontsize=4)


        for i, label1 in enumerate(_labels1):
            axes[1].text(_mkpts1[:, 0][i]-10, _mkpts1[:, 1][i]-10, label1, fontsize=4)
        
        for i, label2 in enumerate(labels2):
            axes[2].text(mkpts2[:, 0][i]-10, mkpts2[:, 1][i]-10, label2, fontsize=4)

        axes[0].text(10.0, 800.0, f'image:{start_idx}', fontsize=12)
        axes[1].text(10.0, 800.0, f'image:{start_idx+1}', fontsize=12 )
        axes[2].text(10.0, 800.0, f'image:{start_idx+2}', fontsize=12 )

    # put txts
    txt_color = 'k'
    text_ = fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=8, va='top', ha='left', color=txt_color)
    
    text_.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0.0, dpi=dpi)
        plt.close()
    else:
        return fig


def make_matching_figure_color_track(
        img0, img1, img2, img3, img4, img5, img6, img7, img8, img9,
        mkpts0, mkpts1, mkpts2, mkpts3, mkpts4, mkpts5, mkpts6, mkpts7, mkpts8, mkpts9,
        _mkpts1, _mkpts2, _mkpts3, _mkpts4, _mkpts5, _mkpts6, _mkpts7, _mkpts8,
        text11, text22, text33, text44, text55, text66, text77, text88,
        bind_list11, bind_list22, bind_list33, bind_list44, bind_list55, bind_list66, bind_list77, bind_list88,
        valid_list11, valid_list22, valid_list33, valid_list44, valid_list55, valid_list66, valid_list77, valid_list88,
        end11, end22, end33, end44, end55, end66, end77, end88,
        kpts0=None, kpts1=None, kpts2=None, kpts3=None, kpts4=None, kpts5=None, kpts6=None, kpts7=None, kpts8=None, kpts9=None,
        dpi=75, path=None, _all=False):
    
    # draw image pair
    """
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    assert mkpts1.shape[0] == mkpts2.shape[0], f'mkpts1: {mkpts1.shape[0]} v.s. mkpts2: {mkpts2.shape[0]}'
    assert mkpts2.shape[0] == mkpts3.shape[0], f'mkpts2: {mkpts2.shape[0]} v.s. mkpts3: {mkpts3.shape[0]}'
    assert mkpts3.shape[0] == mkpts4.shape[0], f'mkpts3: {mkpts3.shape[0]} v.s. mkpts4: {mkpts4.shape[0]}'
    assert mkpts4.shape[0] == mkpts5.shape[0], f'mkpts4: {mkpts4.shape[0]} v.s. mkpts5: {mkpts5.shape[0]}'
    assert mkpts5.shape[0] == mkpts6.shape[0], f'mkpts5: {mkpts5.shape[0]} v.s. mkpts6: {mkpts6.shape[0]}'
    assert mkpts6.shape[0] == mkpts7.shape[0], f'mkpts6: {mkpts6.shape[0]} v.s. mkpts7: {mkpts7.shape[0]}'
    assert mkpts7.shape[0] == mkpts8.shape[0], f'mkpts7: {mkpts7.shape[0]} v.s. mkpts8: {mkpts8.shape[0]}'
    assert mkpts8.shape[0] == mkpts9.shape[0], f'mkpts8: {mkpts8.shape[0]} v.s. mkpts9: {mkpts9.shape[0]}'
    """

    fig, axes = plt.subplots(1, 10, figsize=(30, 12), dpi=dpi)
    im0= axes[0].imshow(img0)
    axes[1].imshow(img1)
    axes[2].imshow(img2)
    axes[3].imshow(img3)
    axes[4].imshow(img4)
    axes[5].imshow(img5)
    axes[6].imshow(img6)
    axes[7].imshow(img7)
    axes[8].imshow(img8)
    axes[9].imshow(img9)
    for i in range(10):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        assert kpts2 is not None

        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=1)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=1)

    # draw matches
    if (mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0) or _all:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fkpts2 = transFigure.transform(axes[2].transData.transform(mkpts2))
        fkpts3 = transFigure.transform(axes[3].transData.transform(mkpts3))
        fkpts4 = transFigure.transform(axes[4].transData.transform(mkpts4))
        fkpts5 = transFigure.transform(axes[5].transData.transform(mkpts5))
        fkpts6 = transFigure.transform(axes[6].transData.transform(mkpts6))
        fkpts7 = transFigure.transform(axes[7].transData.transform(mkpts7))
        fkpts8 = transFigure.transform(axes[8].transData.transform(mkpts8))
        fkpts9 = transFigure.transform(axes[9].transData.transform(mkpts9))

        _fkpts1 = transFigure.transform(axes[1].transData.transform(_mkpts1))
        _fkpts2 = transFigure.transform(axes[2].transData.transform(_mkpts2))
        _fkpts3 = transFigure.transform(axes[3].transData.transform(_mkpts3))
        _fkpts4 = transFigure.transform(axes[4].transData.transform(_mkpts4))
        _fkpts5 = transFigure.transform(axes[5].transData.transform(_mkpts5))
        _fkpts6 = transFigure.transform(axes[6].transData.transform(_mkpts6))
        _fkpts7 = transFigure.transform(axes[7].transData.transform(_mkpts7))
        _fkpts8 = transFigure.transform(axes[8].transData.transform(_mkpts8))

        # draw lines
        lines01 = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, linewidth=0.8, c='cyan')
                                        for i in range(len(mkpts0))]
        
        lines12 = [matplotlib.lines.Line2D((_fkpts1[i, 0], fkpts2[i, 0]),
                                            (_fkpts1[i, 1], fkpts2[i, 1]),
                                            transform=fig.transFigure, c=[_get_color(flag) for flag in valid_list11][i], linewidth=0.8)
                                        for i in range(len(_mkpts1))]

        lines23 = [matplotlib.lines.Line2D((_fkpts2[i, 0], fkpts3[i, 0]),
                                            (_fkpts2[i, 1], fkpts3[i, 1]),
                                            transform=fig.transFigure, c=[_get_color(flag) for flag in valid_list22][i], linewidth=0.8)
                                        for i in range(len(_mkpts2))]
        
        lines34 = [matplotlib.lines.Line2D((_fkpts3[i, 0], fkpts4[i, 0]),
                                            (_fkpts3[i, 1], fkpts4[i, 1]),
                                            transform=fig.transFigure, c=[_get_color(flag) for flag in valid_list33][i], linewidth=0.8)
                                        for i in range(len(_mkpts3))]
        
        lines45 = [matplotlib.lines.Line2D((_fkpts4[i, 0], fkpts5[i, 0]),
                                            (_fkpts4[i, 1], fkpts5[i, 1]),
                                            transform=fig.transFigure, c=[_get_color(flag) for flag in valid_list44][i], linewidth=0.8)
                                        for i in range(len(_mkpts4))]
        
        lines56 = [matplotlib.lines.Line2D((_fkpts5[i, 0], fkpts6[i, 0]),
                                            (_fkpts5[i, 1], fkpts6[i, 1]),
                                            transform=fig.transFigure, c=[_get_color(flag) for flag in valid_list55][i], linewidth=0.8)
                                        for i in range(len(_mkpts5))]
        
        lines67 = [matplotlib.lines.Line2D((_fkpts6[i, 0], fkpts7[i, 0]),
                                            (_fkpts6[i, 1], fkpts7[i, 1]),
                                            transform=fig.transFigure, c=[_get_color(flag) for flag in valid_list66][i], linewidth=0.8)
                                        for i in range(len(_mkpts6))]
        
        lines78 = [matplotlib.lines.Line2D((_fkpts7[i, 0], fkpts8[i, 0]),
                                            (_fkpts7[i, 1], fkpts8[i, 1]),
                                            transform=fig.transFigure, c=[_get_color(flag) for flag in valid_list77][i], linewidth=0.8)
                                        for i in range(len(_mkpts7))]
        
        lines89 = [matplotlib.lines.Line2D((_fkpts8[i, 0], fkpts9[i, 0]),
                                            (_fkpts8[i, 1], fkpts9[i, 1]),
                                            transform=fig.transFigure, c=[_get_color(flag) for flag in valid_list88][i], linewidth=0.8)
                                        for i in range(len(_mkpts8))]

        
        dist_lines11 = [matplotlib.lines.Line2D((_fkpts1[i, 0], fkpts1[t, 0]),
                                            (_fkpts1[i, 1], fkpts1[t, 1]),
                                            transform=fig.transFigure, c= 'green', linewidth=0.4)
                                        for i, t in bind_list11]
        dist_lines22 = [matplotlib.lines.Line2D((_fkpts2[i, 0], fkpts2[t, 0]),
                                            (_fkpts2[i, 1], fkpts2[t, 1]),
                                            transform=fig.transFigure, c= 'green', linewidth=0.4)
                                        for i, t in bind_list22]
        dist_lines33 = [matplotlib.lines.Line2D((_fkpts3[i, 0], fkpts3[t, 0]),
                                            (_fkpts3[i, 1], fkpts3[t, 1]),
                                            transform=fig.transFigure, c= 'green', linewidth=0.4)
                                        for i, t in bind_list33]
        dist_lines44 = [matplotlib.lines.Line2D((_fkpts4[i, 0], fkpts4[t, 0]),
                                            (_fkpts4[i, 1], fkpts4[t, 1]),
                                            transform=fig.transFigure, c= 'green', linewidth=0.4)
                                        for i, t in bind_list44]
        dist_lines55 = [matplotlib.lines.Line2D((_fkpts5[i, 0], fkpts5[t, 0]),
                                            (_fkpts5[i, 1], fkpts5[t, 1]),
                                            transform=fig.transFigure, c= 'green', linewidth=0.4)
                                        for i, t in bind_list55]
        dist_lines66 = [matplotlib.lines.Line2D((_fkpts6[i, 0], fkpts6[t, 0]),
                                            (_fkpts6[i, 1], fkpts6[t, 1]),
                                            transform=fig.transFigure, c= 'green', linewidth=0.4)
                                        for i, t in bind_list66]
        dist_lines77 = [matplotlib.lines.Line2D((_fkpts7[i, 0], fkpts7[t, 0]),
                                            (_fkpts7[i, 1], fkpts7[t, 1]),
                                            transform=fig.transFigure, c= 'green', linewidth=0.4)
                                        for i, t in bind_list77]
        dist_lines88 = [matplotlib.lines.Line2D((_fkpts8[i, 0], fkpts8[t, 0]),
                                            (_fkpts8[i, 1], fkpts8[t, 1]),
                                            transform=fig.transFigure, c= 'green', linewidth=0.4)
                                        for i, t in bind_list88]

        
        fig.lines.extend(lines01)
        fig.lines.extend(lines12)
        fig.lines.extend(lines23)
        fig.lines.extend(lines34)
        fig.lines.extend(lines45)
        fig.lines.extend(lines56)
        fig.lines.extend(lines67)
        fig.lines.extend(lines78)
        fig.lines.extend(lines89)

        upper_limit_y = 960 // 2
        lower_limit_y = 0
        upper_limit_x = 10000
        lower_limit_x = 0

        max_point = 20
        point_count = 0
        cmap = plt.cm.plasma
        colors = cmap(np.linspace(0, 1, len(mkpts0)))
        for _point, color in zip(mkpts0, colors):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[0].plot(_point[0], _point[1], 'o', markersize=2, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts1, _get_marker(valid_list11, 0), [_get_color(flag) for flag in valid_list11]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[1].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts1, _get_marker(valid_list11, 1, end11), _get_color2(valid_list11, end11)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[1].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts2, _get_marker(valid_list22, 0), [_get_color(flag) for flag in valid_list22]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[2].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts2, _get_marker(valid_list22, 1, end22), _get_color2(valid_list22, end22)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[2].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts3, _get_marker(valid_list33, 0), [_get_color(flag) for flag in valid_list33]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[3].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts3, _get_marker(valid_list33, 1, end33), _get_color2(valid_list33, end33)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[3].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts4, _get_marker(valid_list44, 0), [_get_color(flag) for flag in valid_list44]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[4].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts4, _get_marker(valid_list44, 1, end44), _get_color2(valid_list44, end44)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[4].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts5, _get_marker(valid_list55, 0), [_get_color(flag) for flag in valid_list55]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[5].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts5, _get_marker(valid_list55, 1, end55), _get_color2(valid_list55, end55)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[5].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts6, _get_marker(valid_list66, 0), [_get_color(flag) for flag in valid_list66]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[6].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts6, _get_marker(valid_list66, 1, end66), _get_color2(valid_list66, end66)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[6].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts7, _get_marker(valid_list77, 0), [_get_color(flag) for flag in valid_list77]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[7].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts7, _get_marker(valid_list77, 1, end77), _get_color2(valid_list77, end77)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[7].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts8, _get_marker(valid_list88, 0), [_get_color(flag) for flag in valid_list88]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[8].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts8, _get_marker(valid_list88, 1, end88), _get_color2(valid_list88, end88)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[8].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point in mkpts9:
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[9].plot(_point[0], _point[1], 'o', markersize=2, alpha=1.0, c='cyan')
                    point_count += 1



        axes[0].text(10.0, 950.0, f'image:{0}', fontsize=6)
        axes[1].text(10.0, 950.0, f'image:{1}', fontsize=6 )
        axes[2].text(10.0, 950.0, f'image:{2}', fontsize=6 )
        axes[3].text(10.0, 950.0, f'image:{3}', fontsize=6 )
        axes[4].text(10.0, 950.0, f'image:{4}', fontsize=6 )
        axes[5].text(10.0, 950.0, f'image:{5}', fontsize=6 )
        axes[6].text(10.0, 950.0, f'image:{6}', fontsize=6 )
        axes[7].text(10.0, 950.0, f'image:{7}', fontsize=6 )
        axes[8].text(10.0, 950.0, f'image:{8}', fontsize=6 )
        axes[9].text(10.0, 950.0, f'image:{9}', fontsize=6 )

    # put txts
    txt_color = 'k'
    text_11 = fig.text(
        0.01, 0.99, '\n'.join(text11), transform=fig.axes[1].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    text_22 = fig.text(
        0.01, 0.99, '\n'.join(text22), transform=fig.axes[2].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    text_33 = fig.text(
        0.01, 0.99, '\n'.join(text33), transform=fig.axes[3].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    text_44 = fig.text(
        0.01, 0.99, '\n'.join(text44), transform=fig.axes[4].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    text_55 = fig.text(
        0.01, 0.99, '\n'.join(text55), transform=fig.axes[5].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    text_66 = fig.text(
        0.01, 0.99, '\n'.join(text66), transform=fig.axes[6].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    text_77 = fig.text(
        0.01, 0.99, '\n'.join(text77), transform=fig.axes[7].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    text_88 = fig.text(
        0.01, 0.99, '\n'.join(text88), transform=fig.axes[8].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    


    text_11.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    text_22.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    text_33.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    text_44.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    text_55.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    text_66.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    text_77.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    text_88.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])


    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0.0, dpi=dpi)
        plt.close()
    else:
        return fig




def make_matching_figure_color_track_point(
        img0, img1, img2, img3, img4, img5, img6, img7, img8, img9,
        mkpts0, mkpts1, mkpts2, mkpts3, mkpts4, mkpts5, mkpts6, mkpts7, mkpts8, mkpts9,
        _mkpts1, _mkpts2, _mkpts3, _mkpts4, _mkpts5, _mkpts6, _mkpts7, _mkpts8,
        text11, text22, text33, text44, text55, text66, text77, text88,
        bind_list11, bind_list22, bind_list33, bind_list44, bind_list55, bind_list66, bind_list77, bind_list88,
        valid_list11, valid_list22, valid_list33, valid_list44, valid_list55, valid_list66, valid_list77, valid_list88,
        end11, end22, end33, end44, end55, end66, end77, end88,
        kpts0=None, kpts1=None, kpts2=None, kpts3=None, kpts4=None, kpts5=None, kpts6=None, kpts7=None, kpts8=None, kpts9=None,
        dpi=75, path=None, _all=False):
    
    # draw image pair
    """
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    assert mkpts1.shape[0] == mkpts2.shape[0], f'mkpts1: {mkpts1.shape[0]} v.s. mkpts2: {mkpts2.shape[0]}'
    assert mkpts2.shape[0] == mkpts3.shape[0], f'mkpts2: {mkpts2.shape[0]} v.s. mkpts3: {mkpts3.shape[0]}'
    assert mkpts3.shape[0] == mkpts4.shape[0], f'mkpts3: {mkpts3.shape[0]} v.s. mkpts4: {mkpts4.shape[0]}'
    assert mkpts4.shape[0] == mkpts5.shape[0], f'mkpts4: {mkpts4.shape[0]} v.s. mkpts5: {mkpts5.shape[0]}'
    assert mkpts5.shape[0] == mkpts6.shape[0], f'mkpts5: {mkpts5.shape[0]} v.s. mkpts6: {mkpts6.shape[0]}'
    assert mkpts6.shape[0] == mkpts7.shape[0], f'mkpts6: {mkpts6.shape[0]} v.s. mkpts7: {mkpts7.shape[0]}'
    assert mkpts7.shape[0] == mkpts8.shape[0], f'mkpts7: {mkpts7.shape[0]} v.s. mkpts8: {mkpts8.shape[0]}'
    assert mkpts8.shape[0] == mkpts9.shape[0], f'mkpts8: {mkpts8.shape[0]} v.s. mkpts9: {mkpts9.shape[0]}'
    """

    fig, axes = plt.subplots(1, 10, figsize=(30, 12), dpi=dpi)
    im0 = axes[0].imshow(img0)
    axes[1].imshow(img1)
    axes[2].imshow(img2)
    axes[3].imshow(img3)
    axes[4].imshow(img4)
    axes[5].imshow(img5)
    axes[6].imshow(img6)
    axes[7].imshow(img7)
    axes[8].imshow(img8)
    axes[9].imshow(img9)
    for i in range(10):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        assert kpts2 is not None

        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=1)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=1)

    # draw matches
    if (mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0) or _all:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fkpts2 = transFigure.transform(axes[2].transData.transform(mkpts2))
        fkpts3 = transFigure.transform(axes[3].transData.transform(mkpts3))
        fkpts4 = transFigure.transform(axes[4].transData.transform(mkpts4))
        fkpts5 = transFigure.transform(axes[5].transData.transform(mkpts5))
        fkpts6 = transFigure.transform(axes[6].transData.transform(mkpts6))
        fkpts7 = transFigure.transform(axes[7].transData.transform(mkpts7))
        fkpts8 = transFigure.transform(axes[8].transData.transform(mkpts8))
        fkpts9 = transFigure.transform(axes[9].transData.transform(mkpts9))

        _fkpts1 = transFigure.transform(axes[1].transData.transform(_mkpts1))
        _fkpts2 = transFigure.transform(axes[2].transData.transform(_mkpts2))
        _fkpts3 = transFigure.transform(axes[3].transData.transform(_mkpts3))
        _fkpts4 = transFigure.transform(axes[4].transData.transform(_mkpts4))
        _fkpts5 = transFigure.transform(axes[5].transData.transform(_mkpts5))
        _fkpts6 = transFigure.transform(axes[6].transData.transform(_mkpts6))
        _fkpts7 = transFigure.transform(axes[7].transData.transform(_mkpts7))
        _fkpts8 = transFigure.transform(axes[8].transData.transform(_mkpts8))



        upper_limit_y = 960 // 2 - 10
        lower_limit_y = 200
        upper_limit_x = 1280 * 4/4
        lower_limit_x = 1280 * 3/4

        upper_limit_y = 10000
        lower_limit_y = 0
        upper_limit_x = 10000
        lower_limit_x = 0
        max_point = 10000
        point_count = 0
        cmap = plt.cm.plasma
        colors = cmap(np.linspace(0, 1, len(mkpts0)))
        for _point, color in zip(mkpts0, colors):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[0].plot(_point[0], _point[1], 'o', markersize=2, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts1, _get_marker(valid_list11, 0), [_get_color(flag) for flag in valid_list11]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[1].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts1, _get_marker(valid_list11, 1, end11), _get_color2(valid_list11, end11)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[1].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts2, _get_marker(valid_list22, 0), [_get_color(flag) for flag in valid_list22]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[2].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts2, _get_marker(valid_list22, 1, end22), _get_color2(valid_list22, end22)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[2].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts3, _get_marker(valid_list33, 0), [_get_color(flag) for flag in valid_list33]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[3].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts3, _get_marker(valid_list33, 1, end33), _get_color2(valid_list33, end33)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[3].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts4, _get_marker(valid_list44, 0), [_get_color(flag) for flag in valid_list44]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[4].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts4, _get_marker(valid_list44, 1, end44), _get_color2(valid_list44, end44)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[4].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts5, _get_marker(valid_list55, 0), [_get_color(flag) for flag in valid_list55]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[5].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts5, _get_marker(valid_list55, 1, end55), _get_color2(valid_list55, end55)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[5].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts6, _get_marker(valid_list66, 0), [_get_color(flag) for flag in valid_list66]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[6].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts6, _get_marker(valid_list66, 1, end66), _get_color2(valid_list66, end66)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[6].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts7, _get_marker(valid_list77, 0), [_get_color(flag) for flag in valid_list77]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[7].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts7, _get_marker(valid_list77, 1, end77), _get_color2(valid_list77, end77)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[7].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point, marker, color in zip(_mkpts8, _get_marker(valid_list88, 0), [_get_color(flag) for flag in valid_list88]):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[8].plot(_point[0], _point[1], marker=marker, markersize=2, alpha=1.0, c=color)
                    point_count += 1
        point_count = 0
        for _point, marker, color in zip(mkpts8, _get_marker(valid_list88, 1, end88), _get_color2(valid_list88, end88)):
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[8].plot(_point[0], _point[1], marker=marker, markersize=1.5, alpha=1.0, c=color)
                    point_count += 1

        point_count = 0
        for _point in mkpts9:
            if lower_limit_x < _point[0] < upper_limit_x and lower_limit_y < _point[1] < upper_limit_y:
                if point_count < max_point:
                    axes[9].plot(_point[0], _point[1], 'o', markersize=2, alpha=1.0, c='cyan')
                    point_count += 1




    # put txts
    txt_color = 'k'
    text_11 = fig.text(
        0.01, 0.99, '\n'.join(text11), transform=fig.axes[1].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    text_22 = fig.text(
        0.01, 0.99, '\n'.join(text22), transform=fig.axes[2].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    text_33 = fig.text(
        0.01, 0.99, '\n'.join(text33), transform=fig.axes[3].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    text_44 = fig.text(
        0.01, 0.99, '\n'.join(text44), transform=fig.axes[4].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    text_55 = fig.text(
        0.01, 0.99, '\n'.join(text55), transform=fig.axes[5].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    text_66 = fig.text(
        0.01, 0.99, '\n'.join(text66), transform=fig.axes[6].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    text_77 = fig.text(
        0.01, 0.99, '\n'.join(text77), transform=fig.axes[7].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    text_88 = fig.text(
        0.01, 0.99, '\n'.join(text88), transform=fig.axes[8].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
    


    text_11.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    text_22.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    text_33.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    text_44.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    text_55.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    text_66.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    text_77.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])
    text_88.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()])


    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0.0, dpi=dpi)
        plt.close()

    else:
        return fig
    





def make_matching_figure_color_dict(
        img0, img1, img2, img3, img4, img5, img6, img7, img8, img9,
        text11, text22, text33, text44, text55, text66, text77, text88,
        kpts0=None, kpts1=None, kpts2=None, kpts3=None, kpts4=None, kpts5=None, kpts6=None, kpts7=None, kpts8=None, kpts9=None,
        dpi=75, path=None, _all=False, track_dict=None, draw_line=False, img_num=10, is_roi=False, super_point=False,
        data_01=None, data_12=None, data_23=None, data_34=None, data_45=None, data_56=None, data_67=None, data_78=None, data_89=None, dx=0, dy=0):
    
    # draw image pair
    """
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    assert mkpts1.shape[0] == mkpts2.shape[0], f'mkpts1: {mkpts1.shape[0]} v.s. mkpts2: {mkpts2.shape[0]}'
    assert mkpts2.shape[0] == mkpts3.shape[0], f'mkpts2: {mkpts2.shape[0]} v.s. mkpts3: {mkpts3.shape[0]}'
    assert mkpts3.shape[0] == mkpts4.shape[0], f'mkpts3: {mkpts3.shape[0]} v.s. mkpts4: {mkpts4.shape[0]}'
    assert mkpts4.shape[0] == mkpts5.shape[0], f'mkpts4: {mkpts4.shape[0]} v.s. mkpts5: {mkpts5.shape[0]}'
    assert mkpts5.shape[0] == mkpts6.shape[0], f'mkpts5: {mkpts5.shape[0]} v.s. mkpts6: {mkpts6.shape[0]}'
    assert mkpts6.shape[0] == mkpts7.shape[0], f'mkpts6: {mkpts6.shape[0]} v.s. mkpts7: {mkpts7.shape[0]}'
    assert mkpts7.shape[0] == mkpts8.shape[0], f'mkpts7: {mkpts7.shape[0]} v.s. mkpts8: {mkpts8.shape[0]}'
    assert mkpts8.shape[0] == mkpts9.shape[0], f'mkpts8: {mkpts8.shape[0]} v.s. mkpts9: {mkpts9.shape[0]}'
    """
    assert track_dict is not None

    def make_number_overlay(width=832, height=832, cell=8, font=None, font_size=8,
                        color=(255,255,0,220)):
        img = Image.new('RGBA', (width, height), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        if font is None:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        nx = width // cell
        ny = height // cell
        for j in range(ny):
            for i in range(nx):
                t = j*nx + i
                cx = int((i + 0.5) * cell)
                cy = int((j + 0.5) * cell)
                try:
                    draw.text((cx, cy), str(t), fill=color, font=font, anchor='mm')
                except TypeError:
                    w, h = draw.textsize(str(t), font=font)
                    draw.text((cx - w//2, cy - h//2), str(t), fill=color, font=font)
        return np.asarray(img)

    overlay_rgba = make_number_overlay(width=832, height=832, cell=8, font_size=1, color=(255,255,0,220))

    # draw matches
    upper_limit_y = 10000
    lower_limit_y = 0
    upper_limit_x = 10000
    lower_limit_x = 0
    max_point = 500
    if is_roi:
        upper_limit_y = 960 // 2
        lower_limit_y = 280
        upper_limit_x = 1280
        lower_limit_x = 1000

        max_point = 150

        upper_limit_y = 152
        lower_limit_y = 48    
        upper_limit_x = 832     
        lower_limit_x = 600    

        upper_limit_y = 288
        lower_limit_y = 256    
        upper_limit_x = 336     
        lower_limit_x = 248 

        upper_limit_y = 312
        lower_limit_y = 208    
        upper_limit_x = 824     
        lower_limit_x = 728 

    fig, axes = plt.subplots(1, img_num, figsize=(4*img_num, 3*img_num), dpi=dpi)
    axes[0].imshow(img0)
    axes[1].imshow(img1)

    if img2 is not None:
        axes[2].imshow(img2)
    
    if img3 is not None:
        axes[3].imshow(img3)
    
    if img4 is not None:
        axes[4].imshow(img4)
    if img5 is not None:
        axes[5].imshow(img5)
    if img6 is not None:
        axes[6].imshow(img6)
    if img7 is not None:
        axes[7].imshow(img7)
    if img8 is not None:
        axes[8].imshow(img8)
    if img9 is not None:
        axes[9].imshow(img9)
    for i in range(img_num):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
        axes[i].autoscale(False) 
        axes[i].set_xticks(np.arange(0, 833, 104))
        axes[i].set_yticks(np.arange(0, 833, 104))
        axes[i].set_xticks(np.arange(0, 833, 8), minor=True)
        axes[i].set_yticks(np.arange(0, 833, 8), minor=True)
        axes[i].grid(which="major", linestyle="-", linewidth=0.2)
        axes[i].grid(which="minor", linestyle=":", linewidth=0.05)
        
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        assert kpts2 is not None

        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=1)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=1)

    fig.canvas.draw()

    point_count = 0
    plot_track_id_list = []
    out_track_id_list = []
    first_out_list = []
    other_out_list = []
    track_number = 0
    for track_id, track_data in track_dict.items():
        start_id = track_data['start_id']
        end_id   = track_data['end_id']
        points   = track_data['points']  # [(x,y), ...]
        if super_point:
            points_sp = track_data['points_sp']  # [(x,y), ...]
        else:
            points_sp = points
        is_draw = False
        data_pts = {}  # {frame_idx: (x,y)}

        print_pts = {}

        if point_count >= max_point:
            track_number += 1
            continue

        #    continue

        for i, ((x, y), (sp_x,sp_y))  in enumerate(zip(points, points_sp)):
            frame_idx = i + start_id
            print_pts[frame_idx] = (x,y)
            if not (0 <= frame_idx < len(axes)):
                continue

            if not (lower_limit_x <= x <= upper_limit_x and lower_limit_y <= y <= upper_limit_y):
                continue

            if frame_idx == start_id:
                is_draw = True

            if (point_count < max_point) and is_draw:
                if frame_idx == start_id:
                    c = 'green'
                elif frame_idx == end_id:
                    c = 'red'
                else:
                    c = 'cyan'

                if super_point:
                    data_pts[frame_idx] = (sp_x, sp_y)
                    if frame_idx != start_id:
                        if data_pts[frame_idx][0]-data_pts[frame_idx-1][0] == dx and data_pts[frame_idx][1]-data_pts[frame_idx-1][1] == dy:
                            c = 'cyan'
                        else:
                            c = 'yellow'
                    axes[frame_idx].plot(sp_x, sp_y, marker='.', markersize=0.2, alpha=1.0, c=c)
                    axes[frame_idx].text(sp_x-2, sp_y-2, track_id, fontsize=0.3)
                else:
                    data_pts[frame_idx] = (x, y)
                    if frame_idx != start_id:
                        if data_pts[frame_idx][0]-data_pts[frame_idx-1][0] == dx and data_pts[frame_idx][1]-data_pts[frame_idx-1][1] == dy:
                            c = 'cyan'
                        else:
                            c = 'yellow'
                    axes[frame_idx].plot(x, y, marker='.', markersize=0.2, alpha=1.0, c=c)
                    axes[frame_idx].text(x-2, y-2, track_id, fontsize=0.3)




        if draw_line and is_draw:
            for t in range(start_id, end_id):
                if (t in data_pts) and (t+1 in data_pts):
                    (x0, y0) = data_pts[t]
                    (x1, y1) = data_pts[t+1]
                    if x1 - x0 == dx and y1 - y0 == dy:
                        c = 'cyan'
                    else:
                        c = 'yellow'
                        out_track_id_list.append(track_id)
                    con = ConnectionPatch(
                        xyA=(x0, y0), xyB=(x1, y1),
                        coordsA=axes[t].transData,
                        coordsB=axes[t+1].transData,
                        axesA=axes[t], axesB=axes[t+1],
                        color=c, lw=0.1
                    )
                    fig.add_artist(con)
        if is_draw:
            point_count += 1
            plot_track_id_list.append(track_id)
        

        
        for t in range(start_id, end_id):
            if t in print_pts and t+1 in print_pts:
                (x0, y0) = print_pts[t]
                (x1, y1) = print_pts[t+1]
                if x1 - x0 != dx or y1 - y0 != dy:
                    out_track_id_list.append(track_id)
                    if t == 0:
                        first_out_list.append(track_id)
                    else:
                        if track_id not in first_out_list:
                            other_out_list.append(track_id)
        track_number += 1
                
                    

    if super_point:
        if data_01 is not None:
            for (x, y) in data_01['sp0'].cpu().numpy():
                if not (lower_limit_x <= x <= upper_limit_x and lower_limit_y <= y <= upper_limit_y):
                    continue
                axes[0].plot(x, y, marker='x', markersize=0.2, alpha=0.5, c='yellow')
            for (x, y) in data_01['sp1'].cpu().numpy():
                if not (lower_limit_x <= x <= upper_limit_x and lower_limit_y <= y <= upper_limit_y):
                    continue
                axes[1].plot(x, y, marker='x', markersize=0.2, alpha=0.5, c='yellow')
                
        if data_12 is not None:
            for (x, y) in data_12['sp1'].cpu().numpy():
                if not (lower_limit_x <= x <= upper_limit_x and lower_limit_y <= y <= upper_limit_y):
                    continue
                axes[2].plot(x, y, marker='x', markersize=0.2, alpha=0.5, c='yellow')

        if data_23 is not None:
            for (x, y) in data_23['sp1'].cpu().numpy():
                if not (lower_limit_x <= x <= upper_limit_x and lower_limit_y <= y <= upper_limit_y):
                    continue
                axes[3].plot(x, y, marker='x', markersize=0.2, alpha=0.5, c='yellow')
        if data_34 is not None:
            for (x, y) in data_34['sp1'].cpu().numpy():
                if not (lower_limit_x <= x <= upper_limit_x and lower_limit_y <= y <= upper_limit_y):
                    continue
                axes[4].plot(x, y, marker='x', markersize=0.2, alpha=0.5, c='yellow')
        if data_45 is not None:
            for (x, y) in data_45['sp1'].cpu().numpy():
                if not (lower_limit_x <= x <= upper_limit_x and lower_limit_y <= y <= upper_limit_y):
                    continue    
                axes[5].plot(x, y, marker='x', markersize=0.2, alpha=0.5, c='yellow')
        if data_56 is not None:
            for (x, y) in data_56['sp1'].cpu().numpy():
                if not (lower_limit_x <= x <= upper_limit_x and lower_limit_y <= y <= upper_limit_y):
                    continue
                axes[6].plot(x, y, marker='x', markersize=0.2, alpha=0.5, c='yellow')
        if data_67 is not None:
            for (x, y) in data_67['sp1'].cpu().numpy():
                if not (lower_limit_x <= x <= upper_limit_x and lower_limit_y <= y <= upper_limit_y):
                    continue
                axes[7].plot(x, y, marker='x', markersize=0.2, alpha=0.5, c='yellow')
        if data_78 is not None:
            for (x, y) in data_78['sp1'].cpu().numpy():
                if not (lower_limit_x <= x <= upper_limit_x and lower_limit_y <= y <= upper_limit_y):
                    continue
                axes[8].plot(x, y, marker='x', markersize=0.2, alpha=0.5, c='yellow')
        if data_89 is not None:
            for (x, y) in data_89['sp1'].cpu().numpy():
                if not (lower_limit_x <= x <= upper_limit_x and lower_limit_y <= y <= upper_limit_y):
                    continue
                axes[9].plot(x, y, marker='x', markersize=0.2, alpha=0.5, c='yellow')

    # put txts
    txt_color = 'k'
    if text11 is not None:
        text_11 = fig.text(
            0.01, 0.99, '\n'.join(text11), transform=fig.axes[1].transAxes,
            fontsize=6, va='top', ha='left', color=txt_color)
        text_11.set_path_effects([path_effects.Stroke(linewidth=0.1, foreground='white'), path_effects.Normal()])
    
    if text22 is not None:
        text_22 = fig.text(
        0.01, 0.99, '\n'.join(text22), transform=fig.axes[2].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
        text_22.set_path_effects([path_effects.Stroke(linewidth=0.1, foreground='white'), path_effects.Normal()])
    
    if text33 is not None:
        text_33 = fig.text(
            0.01, 0.99, '\n'.join(text33), transform=fig.axes[3].transAxes,
            fontsize=6, va='top', ha='left', color=txt_color)
        text_33.set_path_effects([path_effects.Stroke(linewidth=0.1, foreground='white'), path_effects.Normal()])
    if text44 is not None:
        text_44 = fig.text(
            0.01, 0.99, '\n'.join(text44), transform=fig.axes[4].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
        text_44.set_path_effects([path_effects.Stroke(linewidth=0.1, foreground='white'), path_effects.Normal()])

    if text55 is not None:
        text_55 = fig.text(
            0.01, 0.99, '\n'.join(text55), transform=fig.axes[5].transAxes,
            fontsize=6, va='top', ha='left', color=txt_color)
        text_55.set_path_effects([path_effects.Stroke(linewidth=0.1, foreground='white'), path_effects.Normal()])
    if text66 is not None:
        text_66 = fig.text(
            0.01, 0.99, '\n'.join(text66), transform=fig.axes[6].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
        text_66.set_path_effects([path_effects.Stroke(linewidth=0.1, foreground='white'), path_effects.Normal()])

    if text77 is not None:
        text_77 = fig.text(
            0.01, 0.99, '\n'.join(text77), transform=fig.axes[7].transAxes,
            fontsize=6, va='top', ha='left', color=txt_color)
        text_77.set_path_effects([path_effects.Stroke(linewidth=0.1, foreground='white'), path_effects.Normal()])
    if text88 is not None:
        text_88 = fig.text(
            0.01, 0.99, '\n'.join(text88), transform=fig.axes[8].transAxes,
        fontsize=6, va='top', ha='left', color=txt_color)
        text_88.set_path_effects([path_effects.Stroke(linewidth=0.1, foreground='white'), path_effects.Normal()])
    

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0.0, dpi=dpi)
        plt.close()
    else:
        return fig



def make_evaluation_figure_color(data, b_id, alpha='dynamic', path=None, dpi=150):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)

    img0 = data['imagec_0'][b_id]
    img1 = data['imagec_1'][b_id]
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    img0 = img0 * (imagenet_std[:, None, None].to(img0.device)) + (imagenet_mean[:, None, None].to(img0.device))
    img1 = img1 * (imagenet_std[:, None, None].to(img1.device)) + (imagenet_mean[:, None, None].to(img1.device))
    img0, img1 = img0.detach().permute(1, 2, 0).cpu().numpy(), img1.detach().permute(1, 2, 0).cpu().numpy()
    img0, img1 = np.clip(img0, 0.0, 1.0), np.clip(img1, 0.0, 1.0)

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    R_errs = data['R_errs'][b_id][0]
    t_errs = data['t_errs'][b_id][0]

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)
    runtime = data['runtime']
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'R_errs: {R_errs:.1f}',
        f't_errs: {t_errs:.1f}',
        f'runtime: {runtime:.1f}',
    ]

    # make the figure
    figure = make_matching_figure_color(img0, img1, kpts0, kpts1,
                                  color, text=text, path=path, dpi=dpi)
    return figure


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def coord_trans(u, v):
    rad = np.sqrt(np.square(u) + np.square(v))
    u /= (rad+1e-3)
    v /= (rad+1e-3)
    return u, v

def kp_color(u, v, resolution):
    h, w = resolution
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    xx, yy = coord_trans(xx, yy)
    vis = flow_uv_to_colors(xx, yy)

    color = vis[v.astype(np.int32), u.astype(np.int32)]
    return color

def draw_kp(img, kps, colors):
    for i, kp in enumerate(kps):
        img = cv2.circle(img, (int(kp[1]), int(kp[0])), 3, colors[i].tolist(), -1)
    return img


def vis_matches(image0, image1, kp0, kp1):
    lh, lw = image0.shape[:2]
    rh, rw = image1.shape[:2]
    mask1 = np.logical_and.reduce(np.array((kp0[:,1]>=0, kp0[:,1]<lw, kp0[:,0]>=0, kp0[:,0]<lh)))
    mask2 = np.logical_and.reduce(np.array((kp1[:,1]>=0, kp1[:,1]<rw, kp1[:,0]>=0, kp1[:,0]<rh)))

    mask = np.logical_and.reduce(np.array((mask1, mask2)))
    kp0 = kp0[mask]
    kp1 = kp1[mask]

    color = kp_color(kp0[:,1], kp0[:,0], (lh, lw))

    image0 = draw_kp(image0, kp0, color)
    image1 = draw_kp(image1, kp1, color)

    pad_width = 5
    zero_image = np.zeros([lh, pad_width, 3])
    vis = np.concatenate([image0, zero_image, image1], axis=1)

    return vis


def make_evaluation_figure_wheel(data, b_id=0, path=None, topk=10000):
    b_mask = data['m_bids'] == b_id

    img0 = data['imagec_0'][b_id]
    img1 = data['imagec_1'][b_id]
    img0 = img0 * (imagenet_std[:, None, None].to(img0.device)) + (imagenet_mean[:, None, None].to(img0.device))
    img1 = img1 * (imagenet_std[:, None, None].to(img1.device)) + (imagenet_mean[:, None, None].to(img1.device))
    img0, img1 = img0.permute(1, 2, 0).detach().cpu().numpy() * 255, img1.permute(1, 2, 0).detach().cpu().numpy() * 255
    img0, img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB), cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img0, img1 = img0.round().astype(np.int32), img1.round().astype(np.int32)
    img0 = np.ascontiguousarray(img0)
    img1 = np.ascontiguousarray(img1)

    kpts0 = data['mkpts0_f'][b_mask]
    kpts1 = data['mkpts1_f'][b_mask]
    mconf = data['mconf_f'][b_mask]

    num = len(mconf) if len(mconf) < topk else topk
    idx = torch.topk(mconf, num, 0).indices
    kpts0 = kpts0[idx]
    kpts1 = kpts1[idx]

    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id][[0, 1]]
        kpts1 = kpts1 / data['scale1'][b_id][[0, 1]]

    # make the figure
    kpts_wh_0 = torch.flip(kpts0, [1]).cpu().numpy()
    kpts_wh_1 = torch.flip(kpts1, [1]).cpu().numpy()
    figure = vis_matches(img0, img1, kpts_wh_0, kpts_wh_1)
    cv2.imwrite(path, figure)
    return figure


def make_confidence_figure(data, b_id=0, path=None, dpi=150, topk=10000):
    img0 = data['imagec_0'][b_id]
    img1 = data['imagec_1'][b_id]
    img0 = img0 * (imagenet_std[:, None, None].to(img0.device)) + (imagenet_mean[:, None, None].to(img0.device))
    img1 = img1 * (imagenet_std[:, None, None].to(img1.device)) + (imagenet_mean[:, None, None].to(img1.device))
    img0, img1 = img0.detach().permute(1, 2, 0).cpu().numpy(), img1.detach().permute(1, 2, 0).cpu().numpy()
    img0, img1 = np.clip(img0, 0.0, 1.0), np.clip(img1, 0.0, 1.0)
    
    lh, lw = img0.shape[:2]
    rh, rw = img1.shape[:2]
    num = len(data['mconf_f']) if len(data['mconf_f']) < topk else topk
    idx = torch.topk(data['mconf_f'], num, 0).indices
    kpts0 = data['mkpts0_f'][idx].detach().cpu().numpy()
    kpts1 = data['mkpts1_f'][idx].detach().cpu().numpy()
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]
    score = data['mconf_f'][idx].cpu().numpy()

    # normalize the score to [0, 1]
    score = (score - score.min()) / (score.max() - score.min())
    color = cm.jet(score)

    text = [
        f'#Matches {len(kpts0)}',
    ]
    # make the figure
    fig = make_matching_figure_color(img0, img1, kpts0, kpts1,
                                  color, text=text, path=path, dpi=dpi)
    return fig


def make_confidence_figure_tri(data, data_tri,b_id=0, path=None, dpi=150, topk=10000, start_idx=0, cor_indices=None):
    img0 = data['imagec_0'][b_id]
    img1 = data['imagec_1'][b_id]
    img0 = img0 * (imagenet_std[:, None, None].to(img0.device)) + (imagenet_mean[:, None, None].to(img0.device))
    img1 = img1 * (imagenet_std[:, None, None].to(img1.device)) + (imagenet_mean[:, None, None].to(img1.device))
    img0, img1 = img0.detach().permute(1, 2, 0).cpu().numpy(), img1.detach().permute(1, 2, 0).cpu().numpy()
    img0, img1 = np.clip(img0, 0.0, 1.0), np.clip(img1, 0.0, 1.0)

    img2 = data_tri['imagec_1'][b_id]
    img2 = img2 * (imagenet_std[:, None, None].to(img2.device)) + (imagenet_mean[:, None, None].to(img2.device))
    img2 = img2.detach().permute(1, 2, 0).cpu().numpy()
    img2 = np.clip(img2, 0.0, 1.0)

    num = len(data['mconf_f']) if len(data['mconf_f']) < topk else topk
    idx = torch.topk(data['mconf_f'], num, 0).indices.detach().cpu().numpy() if cor_indices is None else cor_indices
    idx2 = torch.topk(data_tri['mconf_f'], num+50, 0).indices
    kpts0 = data['mkpts0_f1'][idx].detach().cpu().numpy()
    kpts1 = data['mkpts1_f1'][idx].detach().cpu().numpy()
    kpts1_tri = data_tri['mkpts0_f1'].detach().cpu().numpy()

    kpts1_detail = data['mkpts1_f_detail']
    kpts1_detail_fine_window, kpts1_detail_f, kpts1_detail_ref = kpts1_detail[0][idx], kpts1_detail[1][idx], kpts1_detail[2][idx]
    kpts1_tri_detail = data_tri['mkpts0_f_detail']
    kpts1_tri_detail_fine_window, kpts1_tri_detail_f, kpts1_tri_detail_ref = kpts1_tri_detail

    cor_indices = []
    bind_list  = []
    valid_list = []
    founds = 0
    tree = cKDTree(kpts1_tri)
    for t in range(len(kpts1)):
        dist_i = None
        best_kpt = None
        dist, cor_idx = tree.query(kpts1[t], k=1, distance_upper_bound=8*np.sqrt(2))
        if cor_idx < len(kpts1_tri):
            dist_i = cor_idx
            best_kpt = kpts1_tri[cor_idx]
        if dist_i is not None:
            cor_indices.append(dist_i)
            bind_list.append([len(cor_indices)-1, t])
            founds += 1
            valid_list.append(True if dist <= 2*np.sqrt(2) else False)
    valid_num = np.array(valid_list).sum()
    
    for add_idx in idx2.detach().cpu().numpy():
        if len(cor_indices) == num:
            break
        if add_idx not in cor_indices:
            cor_indices.append(add_idx)
            valid_list.append('new')
    assert len(cor_indices) == num
    _kpts1 = data_tri['mkpts0_f1'][cor_indices].detach().cpu().numpy()
    kpts2 = data_tri['mkpts1_f1'][cor_indices].detach().cpu().numpy()

    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    if 'scale0' in data_tri:
        _kpts1 = _kpts1 / data_tri['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts2 = kpts2 / data_tri['scale1'][b_id].cpu().numpy()[[1, 0]]

    # normalize the score to [0, 1]

    text = [
        f'#Matches {len(kpts0)} (founds: {founds}) 2√2px matches: {valid_num}'
    ]
    # make the figure
    fig = make_matching_figure_color_tri(img0, img1, img2, kpts0, kpts1, _kpts1, kpts2, bind_list, valid_list,
                                  text=text, path=path, dpi=dpi, start_idx=start_idx)
    return fig, cor_indices


def make_confidence_figure_track(data_01, data_12, data_23, data_34, data_45, data_56, data_67, data_78, data_89,
                                 b_id=0, dpi=150, topk=10000, mode=0, _all=False, add_mode=0, result_path=None, 
                                 is_origin_img=False, origin_wh=None, new_wh=None, draw_line=False, img_num=10, is_roi=False, dx=0, dy=0):

    img0 = data_01['imagec_0'][b_id]
    img1 = data_01['imagec_1'][b_id]
    img0 = img0 * (imagenet_std[:, None, None].to(img0.device)) + (imagenet_mean[:, None, None].to(img0.device))
    img1 = img1 * (imagenet_std[:, None, None].to(img1.device)) + (imagenet_mean[:, None, None].to(img1.device))
    img0, img1 = img0.detach().permute(1, 2, 0).cpu().numpy(), img1.detach().permute(1, 2, 0).cpu().numpy()
    img0, img1 = np.clip(img0, 0.0, 1.0), np.clip(img1, 0.0, 1.0)

    img2 = None
    img3 = None
    img4 = None
    img5 = None
    img6 = None
    img7 = None
    img8 = None
    img9 = None
    text11 = None
    text22 = None
    text33 = None
    text44 = None
    text55 = None
    text66 = None
    text77 = None
    text88 = None
    if data_12 is not None:
        img2 = data_12['imagec_1'][b_id]
        img2 = img2 * (imagenet_std[:, None, None].to(img2.device)) + (imagenet_mean[:, None, None].to(img2.device))
        img2 = img2.detach().permute(1, 2, 0).cpu().numpy()
        img2 = np.clip(img2, 0.0, 1.0)

    if data_23 is not None:
        img3 = data_23['imagec_1'][b_id]
        img3 = img3 * (imagenet_std[:, None, None].to(img3.device)) + (imagenet_mean[:, None, None].to(img3.device))
        img3 = img3.detach().permute(1, 2, 0).cpu().numpy()
        img3 = np.clip(img3, 0.0, 1.0)

    if data_34 is not None:
        img4 = data_34['imagec_1'][b_id]
        img4 = img4 * (imagenet_std[:, None, None].to(img4.device)) + (imagenet_mean[:, None, None].to(img4.device))
        img4 = img4.detach().permute(1, 2, 0).cpu().numpy()
        img4 = np.clip(img4, 0.0, 1.0)

    if data_45 is not None:
        img5 = data_45['imagec_1'][b_id]
        img5 = img5 * (imagenet_std[:, None, None].to(img5.device)) + (imagenet_mean[:, None, None].to(img5.device))
        img5 = img5.detach().permute(1, 2, 0).cpu().numpy()
        img5 = np.clip(img5, 0.0, 1.0)
    
    if data_56 is not None:
        img6 = data_56['imagec_1'][b_id]
        img6 = img6 * (imagenet_std[:, None, None].to(img6.device)) + (imagenet_mean[:, None, None].to(img6.device))
        img6 = img6.detach().permute(1, 2, 0).cpu().numpy()
        img6 = np.clip(img6, 0.0, 1.0)

    if data_67 is not None:
        img7 = data_67['imagec_1'][b_id]
        img7 = img7 * (imagenet_std[:, None, None].to(img7.device)) + (imagenet_mean[:, None, None].to(img7.device))
        img7 = img7.detach().permute(1, 2, 0).cpu().numpy()
        img7 = np.clip(img7, 0.0, 1.0)

    if data_78 is not None:
        img8 = data_78['imagec_1'][b_id]
        img8 = img8 * (imagenet_std[:, None, None].to(img8.device)) + (imagenet_mean[:, None, None].to(img8.device))
        img8 = img8.detach().permute(1, 2, 0).cpu().numpy()
        img8 = np.clip(img8, 0.0, 1.0)
    
    if data_89 is not None:
        img9 = data_89['imagec_1'][b_id]
        img9 = img9 * (imagenet_std[:, None, None].to(img9.device)) + (imagenet_mean[:, None, None].to(img9.device))
        img9 = img9.detach().permute(1, 2, 0).cpu().numpy()
        img9 = np.clip(img9, 0.0, 1.0)


    if is_origin_img:
        import torch.nn.functional as F
        from torchvision import transforms
        origin_whs = origin_wh
        new_whs = new_wh
        assert len(origin_whs) == len(new_whs)

        def _restore_img(img, mask, idx=0):
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            img = img[F.interpolate(mask.unsqueeze(0).float(), mode='nearest', size=(832, 832))[0,0].bool().cpu().numpy().astype(bool)].reshape(new_whs[idx][1], new_whs[idx][0], 3)
            img = transforms.functional.resize(transforms.ToPILImage()(img), size=(origin_whs[idx][1], origin_whs[idx][0]), interpolation=transforms.InterpolationMode.BICUBIC)
            return np.array(img)


        img0 = _restore_img(img0, data_01['mask0'], idx=0)
        img1 = _restore_img(img1, data_01['mask1'], idx=1)

        if img2 is not None:
            img2 = _restore_img(img2, data_12['mask1'], idx=2)
        
        if img3 is not None:
            img3 = _restore_img(img3, data_23['mask1'], idx=3)
        
        if img4 is not None:
            img4 = _restore_img(img4, data_34['mask1'], idx=4)
        if img5 is not None:
            img5 = _restore_img(img5, data_45['mask1'], idx=5)
        if img6 is not None:
            img6 = _restore_img(img6, data_56['mask1'], idx=6)
        if img7 is not None:
            img7 = _restore_img(img7, data_67['mask1'], idx=7)
        if img8 is not None:
            img8 = _restore_img(img8, data_78['mask1'], idx=8)
        if img9 is not None:
            img9 = _restore_img(img9, data_89['mask1'], idx=9)


    num = len(data_01['mconf_f']) if len(data_01['mconf_f']) < topk else topk
    idx = torch.topk(data_01['mconf_f'], num, 0).indices
    mconf01 = data_01['mconf_f'][idx].cpu().numpy()
    kpts0 = data_01['mkpts0_f'][idx].detach().cpu().numpy()
    kpts1 = data_01['mkpts1_f'][idx].detach().cpu().numpy()
    kpts0_f_windows, kpts0_coarse_4s, kpts0_subrefs = data_01['mkpts0_f1_window'].detach().cpu().numpy(), data_01['mkpts0_f1_fine'].detach().cpu().numpy(), data_01['mkpts0_subref'].detach().cpu().numpy()
    kpts1_f_windows, kpts1_coarse_4s, kpts1_subrefs = data_01['mkpts1_f1_window'].detach().cpu().numpy(), data_01['mkpts1_f1_fine'].detach().cpu().numpy(), data_01['mkpts1_subref'].detach().cpu().numpy()
    kpts0_f_window, kpts0_coarse_4, kpts0_subref = kpts0_f_windows[idx.detach().cpu().numpy()], kpts0_coarse_4s[idx.detach().cpu().numpy()], kpts0_subrefs[idx.detach().cpu().numpy()]
    kpts1_f_window, kpts1_coarse_4, kpts1_subref = kpts1_f_windows[idx.detach().cpu().numpy()], kpts1_coarse_4s[idx.detach().cpu().numpy()], kpts1_subrefs[idx.detach().cpu().numpy()]


    if data_12 is not None:
        kpts1_search = data_12['mkpts0_f'].detach().cpu().numpy()

    if data_23 is not None:
        kpts2_search = data_23['mkpts0_f'].detach().cpu().numpy()
    
    if data_34 is not None:
        kpts3_search = data_34['mkpts0_f'].detach().cpu().numpy()
    
    if data_45 is not None:
        kpts4_search = data_45['mkpts0_f'].detach().cpu().numpy()

    if data_56 is not None:
        kpts5_search = data_56['mkpts0_f'].detach().cpu().numpy()
    
    if data_67 is not None: 
        kpts6_search = data_67['mkpts0_f'].detach().cpu().numpy()
    
    if data_78 is not None:
        kpts7_search = data_78['mkpts0_f'].detach().cpu().numpy()
    
    if data_89 is not None:
        kpts8_search = data_89['mkpts0_f'].detach().cpu().numpy()

    dist_thr = 0

    if result_path:
        track_json = {}
        track_id = 0

        for kpt0, kpt1, kpt0_coarse_4, kpt0_f_window, kpt0_subref, kpt1_coarse_4, kpt1_f_window, kpt1_subref in zip(kpts0, kpts1, kpts0_coarse_4, kpts0_f_window, kpts0_subref, kpts1_coarse_4, kpts1_f_window, kpts1_subref):

            track_json[track_id] = {
                "start_id": 0,
                "end_id" : img_num - 1,
                "points": [kpt0.copy(), kpt1.copy()],
                "points_detail": [[kpt0_coarse_4.copy().tolist(), kpt0_f_window.copy().tolist(), kpt0_subref.copy().tolist(), int(kpt0_coarse_4.copy().tolist()[0]/4+kpt0_coarse_4.copy().tolist()[1]/4*104)], [kpt1_coarse_4.copy().tolist(), kpt1_f_window.copy().tolist(), kpt1_subref.copy().tolist(), int(kpt1_coarse_4.copy().tolist()[0]/4+kpt1_coarse_4.copy().tolist()[1]/4*104)]],
                "diff" : [],
                "diff_points" : []
            }
            track_id += 1
        
        end_flag = False
        if data_12 is not None and not end_flag:
            _kpts1, kpts2, bind_list11, valid_list11, text11, end11, track_json, end_flag = _find_valid_match_points2_fast(kpts1_search, track_json, data_12, num, b_id, _all, add_mode, dist_thr, 1, img_num, end_flag)

        if data_23 is not None and not end_flag:
            _kpts2, kpts3, bind_list22, valid_list22, text22, end22, track_json, end_flag = _find_valid_match_points2_fast(kpts2_search, track_json, data_23, num, b_id, _all, add_mode, dist_thr, 2, img_num, end_flag)

        if data_34 is not None and not end_flag:
            _kpts3, kpts4, bind_list33, valid_list33, text33, end33, track_json, end_flag = _find_valid_match_points2_fast(kpts3_search, track_json, data_34, num, b_id, _all, add_mode, dist_thr, 3, img_num, end_flag)

        if data_45 is not None and not end_flag:
            _kpts4, kpts5, bind_list44, valid_list44, text44, end44, track_json, end_flag = _find_valid_match_points2_fast(kpts4_search, track_json, data_45, num, b_id, _all, add_mode, dist_thr, 4, img_num, end_flag)

        if data_56 is not None and not end_flag:
            _kpts5, kpts6, bind_list55, valid_list55, text55, end55, track_json, end_flag = _find_valid_match_points2_fast(kpts5_search, track_json, data_56, num, b_id, _all, add_mode, dist_thr, 5, img_num, end_flag)

        if data_67 is not None and not end_flag:
            _kpts6, kpts7, bind_list66, valid_list66, text66, end66, track_json, end_flag = _find_valid_match_points2_fast(kpts6_search, track_json, data_67, num, b_id, _all, add_mode, dist_thr, 6, img_num, end_flag)

        if data_78 is not None and not end_flag:
            _kpts7, kpts8, bind_list77, valid_list77, text77, end77, track_json, end_flag = _find_valid_match_points2_fast(kpts7_search, track_json, data_78, num, b_id, _all, add_mode, dist_thr, 7, img_num, end_flag)

        if data_89 is not None and not end_flag:
            _kpts8, kpts9, bind_list88, valid_list88, text88, end88, track_json, end_flag = _find_valid_match_points2_fast(kpts8_search, track_json, data_89, num, b_id, _all, add_mode, dist_thr, 8, img_num, end_flag)
    else:
        _kpts1, kpts2, bind_list11, valid_list11, text11, end11 = _find_valid_match_points(kpts1_search, kpts1, data_12, num, b_id, _all, add_mode, dist_thr)
        _kpts2, kpts3, bind_list22, valid_list22, text22, end22 = _find_valid_match_points(kpts2_search, kpts2, data_23, num, b_id, _all, add_mode, dist_thr)
        _kpts3, kpts4, bind_list33, valid_list33, text33, end33 = _find_valid_match_points(kpts3_search, kpts3, data_34, num, b_id, _all, add_mode, dist_thr)
        _kpts4, kpts5, bind_list44, valid_list44, text44, end44 = _find_valid_match_points(kpts4_search, kpts4, data_45, num, b_id, _all, add_mode, dist_thr)
        _kpts5, kpts6, bind_list55, valid_list55, text55, end55 = _find_valid_match_points(kpts5_search, kpts5, data_56, num, b_id, _all, add_mode, dist_thr)
        _kpts6, kpts7, bind_list66, valid_list66, text66, end66 = _find_valid_match_points(kpts6_search, kpts6, data_67, num, b_id, _all, add_mode, dist_thr)
        _kpts7, kpts8, bind_list77, valid_list77, text77, end77 = _find_valid_match_points(kpts7_search, kpts7, data_78, num, b_id, _all, add_mode, dist_thr)
        _kpts8, kpts9, bind_list88, valid_list88, text88, end88 = _find_valid_match_points(kpts8_search, kpts8, data_89, num, b_id, _all, add_mode, dist_thr)

    if 'scale0' in data_01:
        kpts0 = kpts0 / data_01['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data_01['scale1'][b_id].cpu().numpy()[[1, 0]]

    if is_origin_img:
        def _restore_kpts(kpts, idx=0):
            kpts[:, 0] = kpts[: ,0] * origin_whs[idx][0] / new_whs[idx][0]
            kpts[:, 1] = kpts[: ,1] * origin_whs[idx][1] / new_whs[idx][1]
            return kpts

        kpts0 = _restore_kpts(kpts0, idx=0)
        kpts1 = _restore_kpts(kpts1, idx=1)
        if data_12 is not None:
            kpts2 = _restore_kpts(kpts2, idx=2)
            _kpts1 = _restore_kpts(_kpts1, idx=1)

        if data_23 is not None:
            kpts3 = _restore_kpts(kpts3, idx=3)
            _kpts2 = _restore_kpts(_kpts2, idx=2)
        
        if data_34 is not None:
            kpts4 = _restore_kpts(kpts4, idx=4)
            _kpts3 = _restore_kpts(_kpts3, idx=3)
        
        if data_45 is not None:
            kpts5 = _restore_kpts(kpts5, idx=5)
            _kpts4 = _restore_kpts(_kpts4, idx=4)
        
        if data_56 is not None:
            kpts6 = _restore_kpts(kpts6, idx=6)
            _kpts5 = _restore_kpts(_kpts5, idx=5)
        if data_67 is not None:
            kpts7 = _restore_kpts(kpts7, idx=7)
            _kpts6 = _restore_kpts(_kpts6, idx=6)

        if data_78 is not None:
            kpts8 = _restore_kpts(kpts8, idx=8)
            _kpts7 = _restore_kpts(_kpts7, idx=7)
        if data_89 is not None:
            kpts9 = _restore_kpts(kpts9, idx=9)
            _kpts8 = _restore_kpts(_kpts8, idx=8)

        for track_id, track in track_json.items():
            track['points'] = _restore_kpts(np.array(track['points'])).tolist()
            if 'diff_points' in track and len(track['diff_points']) > 0:
                restored_diff_points = []
                for pair in track['diff_points']:
                    pair = np.array(pair)  # shape (2, 2)
                    restored_pair = _restore_kpts(pair).tolist()
                    restored_diff_points.append(restored_pair)
                track['diff_points'] = restored_diff_points

        
    
    if result_path:
        import json
        for track_id, track in track_json.items():
            track['points'] = np.array(track['points']).tolist()
            track['diff'] = np.array(track['diff']).tolist()
            track['diff_points'] = np.array(track['diff_points']).tolist()
        with open(result_path, 'w') as f:
            json.dump(track_json, f, indent=2)

    # make the figure
    make_figure = False
    if make_figure:
        if mode == 0 :
            fig = make_matching_figure_color_track(img0, img1, img2, img3, img4, img5, img6, img7, img8, img9, 
                                           kpts0, kpts1, kpts2, kpts3, kpts4, kpts5, kpts6, kpts7, kpts8, kpts9,
                                           _kpts1, _kpts2, _kpts3, _kpts4, _kpts5, _kpts6, _kpts7, _kpts8,
                                           text11, text22, text33, text44, text55, text66, text77, text88,
                                           bind_list11, bind_list22, bind_list33, bind_list44, bind_list55, bind_list66, bind_list77, bind_list88,
                                           valid_list11, valid_list22, valid_list33, valid_list44, valid_list55, valid_list66, valid_list77, valid_list88,
                                           end11, end22, end33, end44, end55, end66, end77, end88,
                                           dpi=dpi, _all=_all)
        elif mode == 1:
            fig = make_matching_figure_color_track_point(img0, img1, img2, img3, img4, img5, img6, img7, img8, img9, 
                                           kpts0, kpts1, kpts2, kpts3, kpts4, kpts5, kpts6, kpts7, kpts8, kpts9,
                                           _kpts1, _kpts2, _kpts3, _kpts4, _kpts5, _kpts6, _kpts7, _kpts8,
                                           text11, text22, text33, text44, text55, text66, text77, text88,
                                           bind_list11, bind_list22, bind_list33, bind_list44, bind_list55, bind_list66, bind_list77, bind_list88,
                                           valid_list11, valid_list22, valid_list33, valid_list44, valid_list55, valid_list66, valid_list77, valid_list88,
                                           end11, end22, end33, end44, end55, end66, end77, end88,
                                           dpi=dpi, _all=_all)
        else:
            fig = make_matching_figure_color_dict(img0, img1, img2, img3, img4, img5, img6, img7, img8, img9,
                                              text11, text22, text33, text44, text55, text66, text77, text88,
                                              dpi=dpi, _all=_all, track_dict=track_json, draw_line=draw_line, img_num=img_num, is_roi=is_roi,
                                              data_01=data_01, data_12=data_12, data_23=data_23, data_34=data_34, data_45=data_45, data_56=data_56, data_67=data_67, data_78=data_78, data_89=data_89, dx=dx, dy=dy)
        return fig


def _find_valid_match_points(kpts_search, origin_kpts, search_data, kpts_num, b_id, _all, add_mode, dist_thr=2*np.sqrt(2)):

    valid_indices = []
    bind_list  = []
    valid_list = []
    end_indices = []
    add_indices = []
    founds = 0
    for t in range(len(origin_kpts)):
        tree = cKDTree(kpts_search)
        dist_i = None
        best_kpt = None
        dist, cor_idx = tree.query(origin_kpts[t], k=1, distance_upper_bound=10000*np.sqrt(2))
        if cor_idx < len(kpts_search):
            dist_i = cor_idx
            best_kpt = kpts_search[cor_idx]
        if (dist_i is not None) and (dist <= dist_thr):
            valid_indices.append(dist_i)
            bind_list.append([len(valid_indices)-1, t])
            founds += 1
            valid_list.append(True if dist <= 0 else False)
            kpts_search[dist_i] = np.array([100000, 100000])  # remove the matched point from search
        else:
            end_indices.append(t)
            if add_mode == 1:
                if dist_i is not None:
                    add_indices.append(dist_i)
                    kpts_search[dist_i] = np.array([100000, 100000])  # remove the matched point from search

    valid_num = np.array(valid_list).sum()
    
    if _all:
        kpts_num = len(search_data['mconf_f']) if len(search_data['mconf_f']) < kpts_num else kpts_num
    if add_mode == 0:
        for add_idx in torch.topk(search_data['mconf_f'], len(search_data['mconf_f']) if len(search_data['mconf_f']) < kpts_num else kpts_num, 0).indices.detach().cpu().numpy():
            if len(valid_indices) == kpts_num:
                break
            if add_idx not in valid_indices:
                valid_indices.append(add_idx)
                valid_list.append('new')
    elif add_mode == 1:
        for add_idx in add_indices:
            if len(valid_indices) == kpts_num:
                break
            if add_idx not in valid_indices:
                valid_indices.append(add_idx)
                valid_list.append('new')
    else:
        pass
    if not _all and (add_mode == 0 or add_mode == 1):
        assert len(valid_indices) == kpts_num

    _kpts1 = search_data['mkpts0_f'][valid_indices].detach().cpu().numpy()
    kpts2 = search_data['mkpts1_f'][valid_indices].detach().cpu().numpy()
    mconf01 = search_data['mconf_f'][valid_indices].cpu().numpy()

    if 'scale0' in search_data:
        _kpts1 = _kpts1 / search_data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts2 = kpts2 / search_data['scale1'][b_id].cpu().numpy()[[1, 0]]
    
    text = [
        f'#Matches {kpts_num} (founds: {founds}) 0px matches: {valid_num}'
    ]

    return _kpts1, kpts2, bind_list, valid_list, text, end_indices


def _find_valid_match_points2(kpts_search, track_json, search_data, kpts_num, b_id, _all, add_mode, dist_thr=2*np.sqrt(2), start_id=0, img_num=10, end_flag=False):

    valid_indices = []
    bind_list  = []
    valid_list = []
    end_indices = []
    add_indices = []
    founds = 0
    end_number = 0

    for t, (track_id, track) in enumerate(track_json.items()):
        end_id = track['end_id']
        if end_id != img_num-1:
            end_number += 1
            continue
        origin_kpt = track['points'][-1]
    
        tree = cKDTree(kpts_search)
        dist_i = None
        best_kpt = None
        best_conf = -100
        best_dist = 10000*np.sqrt(2)
        dists, cor_idxs = tree.query(origin_kpt, k=10, distance_upper_bound=10000*np.sqrt(2))
        if cor_idxs[0] < len(kpts_search):
            dist_i = cor_idxs[0]
            best_dist = dists[0]
            best_conf = search_data['mconf_f'][dist_i].cpu().numpy()
        else:
            dist_i = None

        for dist, cor_idx in zip(dists[1:], cor_idxs[1:]):
            if cor_idx < len(kpts_search):
                _conf = search_data['mconf_f'][cor_idx].cpu().numpy()
                if dist <= best_dist:
                    best_dist = dist
                    if _conf > best_conf:
                        best_conf = _conf
                        dist_i = cor_idx
        if (dist_i is not None) and (best_dist <= dist_thr):
            valid_indices.append(dist_i)
            bind_list.append([len(valid_indices)-1, t-end_number])
            founds += 1
            valid_list.append(True if best_dist <= 0 else False)
            kpts_search[dist_i] = np.array([100000, 100000])  # remove the matched point from search
            new_kpt = search_data['mkpts1_f'][dist_i].detach().cpu().numpy()
            track_json[track_id]['points'].append(new_kpt)

            detail_f_window, detail_coarse_4, detail_subref = search_data['mkpts1_f1_window'].detach().cpu().numpy(), search_data['mkpts1_f1_fine'].detach().cpu().numpy(), search_data['mkpts1_subref'].detach().cpu().numpy()
            track_json[track_id]['points_detail'].append([detail_coarse_4[dist_i].tolist(), detail_f_window[dist_i].tolist(), detail_subref[dist_i].tolist(), int(detail_coarse_4[dist_i].tolist()[0]/4+detail_coarse_4[dist_i].tolist()[1]/4*104)])
            if 'diff' in search_data:
                track_json[track_id]['diff'].append([search_data['diff'][dist_i].tolist()])
            if 'diff_points' in search_data:
                track_json[track_id]['diff_points'].append([search_data['diff_points']['0'][dist_i].tolist(), search_data['diff_points']['1'][dist_i].tolist()])
        else:
            end_indices.append(t-end_number)
            track_json[track_id]['end_id'] = track_json[track_id]['start_id'] + len(track['points']) - 1
            if add_mode == 1:
                if dist_i is not None:
                    add_indices.append(dist_i)
                    kpts_search[dist_i] = np.array([100000, 100000])  # remove the matched point from search

    valid_num = np.array(valid_list).sum()
    
    track_id += 1
    if _all:
        kpts_num = len(search_data['mconf_f']) if len(search_data['mconf_f']) > kpts_num else kpts_num
    if add_mode == 0:
        for add_idx in torch.topk(search_data['mconf_f'], len(search_data['mconf_f']) if len(search_data['mconf_f']) < kpts_num else kpts_num, 0).indices.detach().cpu().numpy():
            if len(valid_indices) == kpts_num:
                break
            if add_idx not in valid_indices:
                valid_indices.append(add_idx)
                valid_list.append('new')
                kpts0_f_window, kpts0_coarse_4, kpts0_subref = search_data['mkpts0_f1_window'].detach().cpu().numpy(), search_data['mkpts0_f1_fine'].detach().cpu().numpy(), search_data['mkpts0_subref'].detach().cpu().numpy()
                kpts1_f_window, kpts1_coarse_4, kpts1_subref = search_data['mkpts1_f1_window'].detach().cpu().numpy(), search_data['mkpts1_f1_fine'].detach().cpu().numpy(), search_data['mkpts1_subref'].detach().cpu().numpy()
                track_json[track_id] = {"start_id": start_id, 
                                        "end_id": img_num - 1,
                                        "points" : [search_data['mkpts0_f'][add_idx].detach().cpu().numpy(), search_data['mkpts1_f'][add_idx].detach().cpu().numpy()],
                                        "points_detail": [[kpts0_coarse_4[add_idx].tolist(), kpts0_f_window[add_idx].tolist(), kpts0_subref[add_idx].tolist(), int(kpts0_coarse_4[add_idx].tolist()[0]/4+kpts0_coarse_4[add_idx].tolist()[1]/4*104)], [kpts1_coarse_4[add_idx].tolist(), kpts1_f_window[add_idx].tolist(), kpts1_subref[add_idx].tolist(), int(kpts1_coarse_4[add_idx].tolist()[0]/4+kpts1_coarse_4[add_idx].tolist()[1]/4*104)]],
                                        "diff" : [],
                                        "diff_points" : []
                }
                track_id +=1

    elif add_mode == 1:
        for add_idx in add_indices:
            if len(valid_indices) == kpts_num:
                break
            if add_idx not in valid_indices:
                valid_indices.append(add_idx)
                valid_list.append('new')
    else:
        pass
    if not _all and (add_mode == 0 or add_mode == 1):
        assert len(valid_indices) == kpts_num

    _kpts1 = search_data['mkpts0_f'][valid_indices].detach().cpu().numpy()
    kpts2 = search_data['mkpts1_f'][valid_indices].detach().cpu().numpy()
    mconf01 = search_data['mconf_f'][valid_indices].cpu().numpy()
    if mconf01.size == 0:
        end_flag = True

    if 'scale0' in search_data:
        _kpts1 = _kpts1 / search_data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts2 = kpts2 / search_data['scale1'][b_id].cpu().numpy()[[1, 0]]
    
    text = [
        f'#Matches {t-end_number+1} (founds: {founds}) 0px matches: {valid_num}'
    ]

    return _kpts1, kpts2, bind_list, valid_list, text, end_indices, track_json, end_flag


def _find_valid_match_points2_fast(kpts_search_np, track_json, search_data, kpts_num, b_id, _all, add_mode,
                                   dist_thr=2*np.sqrt(2), start_id=0, img_num=10, end_flag=False):
    active_ids, origins = [], []
    for tid, tr in track_json.items():
        if tr['end_id'] == img_num - 1:
            active_ids.append(tid)
            origins.append(tr['points'][-1])
    if not origins:
        return None, None, [], [], ["#Matches 0 (founds: 0) 0px matches: 0"], [], track_json, True

    origins = np.asarray(origins)                  # (M, 2)
    kpts_search = np.asarray(kpts_search_np)       # (Ns, 2)
    tree = cKDTree(kpts_search)

    K = 10
    dists, idxs = tree.query(origins, k=min(K, len(kpts_search)), distance_upper_bound=1e9)  # (M,K)
    if dists.ndim == 1: dists, idxs = dists[:, None], idxs[:, None]

    conf = search_data['mconf_f'].detach().cpu().numpy()  # (Ns,)
    conf_mat = np.where(idxs < len(conf), conf[idxs], -1e9)  # (M,K)

    valid_mat = dists <= dist_thr
    score = np.where(valid_mat, conf_mat, -1e9)
    best_j = np.argmax(score, axis=1)
    best_idx = idxs[np.arange(len(origins)), best_j]
    best_dist = dists[np.arange(len(origins)), best_j]

    ok_mask = (best_idx < len(kpts_search)) & (best_dist <= dist_thr)
    chosen_search = best_idx[ok_mask].astype(np.int64)
    chosen_tids   = np.asarray(active_ids, dtype=object)[ok_mask]

    if chosen_search.size > 0:
        keep = []
        seen = {}
        for tid, si in zip(chosen_tids, chosen_search):
            c = conf[si]
            if (si not in seen) or (c > seen[si][0]):
                seen[si] = (c, tid)
        for si, (_, tid) in seen.items():
            keep.append((tid, si))
        chosen_tids  = np.array([t for t,_ in keep], dtype=object)
        chosen_search = np.array([s for _,s in keep], dtype=np.int64)

    if chosen_search.size > 0:
        k1 = search_data['mkpts1_f'][chosen_search].detach().cpu().numpy()
        win1 = search_data['mkpts1_f1_window'].detach().cpu().numpy()[chosen_search]
        c41  = search_data['mkpts1_f1_fine'].detach().cpu().numpy()[chosen_search]
        sub1 = search_data['mkpts1_subref'].detach().cpu().numpy()[chosen_search]

        for (tid, si), p1, w1, f1, s1 in zip(zip(chosen_tids, chosen_search), k1, win1, c41, sub1):
            tr = track_json[tid]
            tr['points'].append(p1)
            tr['points_detail'].append([f1.tolist(), w1.tolist(), s1.tolist(), int(f1[0]/4 + f1[1]/4*104)])
            if 'diff' in search_data:
                tr['diff'].append([search_data['diff'][si].tolist()])
            if 'diff_points' in search_data:
                tr['diff_points'].append([ search_data['diff_points']['0'][si].tolist(),
                                           search_data['diff_points']['1'][si].tolist() ])

    not_ok = [i for i, tid in enumerate(active_ids) if tid not in set(chosen_tids)]
    for i in not_ok:
        tid = active_ids[i]
        tr = track_json[tid]
        tr['end_id'] = tr['start_id'] + len(tr['points']) - 1

    text = [f"#Matches {len(origins)} (founds: {len(chosen_tids)}) 0px matches: {(best_dist[ok_mask]<=0).sum()}"]
    end_flag = (len(chosen_tids) == 0)
    _kpts1 = search_data['mkpts0_f'][chosen_search].detach().cpu().numpy() if chosen_search.size else np.zeros((0,2))
    kpts2  = search_data['mkpts1_f'][chosen_search].detach().cpu().numpy() if chosen_search.size else np.zeros((0,2))
    bind_list = [[i,i] for i in range(len(chosen_search))]
    valid_list = (best_dist[ok_mask] <= 0).tolist()
    end_indices = not_ok
    return _kpts1, kpts2, bind_list, valid_list, text, end_indices, track_json, end_flag



def make_confidence_figure_track_compare(data_01, data_02, data_03, data_04, data_05, data_06, data_07, data_08, data_09,
                                         b_id=0, topk=15000, result_path=None, is_origin_img=True, origin_wh=None, new_wh=None):
    img0 = data_01['imagec_0'][b_id]
    img1 = data_01['imagec_1'][b_id]
    img0 = img0 * (imagenet_std[:, None, None].to(img0.device)) + (imagenet_mean[:, None, None].to(img0.device))
    img1 = img1 * (imagenet_std[:, None, None].to(img1.device)) + (imagenet_mean[:, None, None].to(img1.device))
    img0, img1 = img0.detach().permute(1, 2, 0).cpu().numpy(), img1.detach().permute(1, 2, 0).cpu().numpy()
    img0, img1 = np.clip(img0, 0.0, 1.0), np.clip(img1, 0.0, 1.0)

    img2 = None
    img3 = None
    img4 = None
    img5 = None
    img6 = None
    img7 = None
    img8 = None
    img9 = None
    if data_02 is not None:
        img2 = data_02['imagec_1'][b_id]
        img2 = img2 * (imagenet_std[:, None, None].to(img2.device)) + (imagenet_mean[:, None, None].to(img2.device))
        img2 = img2.detach().permute(1, 2, 0).cpu().numpy()
        img2 = np.clip(img2, 0.0, 1.0)

    if data_03 is not None:
        img3 = data_03['imagec_1'][b_id]
        img3 = img3 * (imagenet_std[:, None, None].to(img3.device)) + (imagenet_mean[:, None, None].to(img3.device))
        img3 = img3.detach().permute(1, 2, 0).cpu().numpy()
        img3 = np.clip(img3, 0.0, 1.0)

    if data_04 is not None:
        img4 = data_04['imagec_1'][b_id]
        img4 = img4 * (imagenet_std[:, None, None].to(img4.device)) + (imagenet_mean[:, None, None].to(img4.device))
        img4 = img4.detach().permute(1, 2, 0).cpu().numpy()
        img4 = np.clip(img4, 0.0, 1.0)

    if data_05 is not None:
        img5 = data_05['imagec_1'][b_id]
        img5 = img5 * (imagenet_std[:, None, None].to(img5.device)) + (imagenet_mean[:, None, None].to(img5.device))
        img5 = img5.detach().permute(1, 2, 0).cpu().numpy()
        img5 = np.clip(img5, 0.0, 1.0)

    if data_06 is not None:
        img6 = data_06['imagec_1'][b_id]
        img6 = img6 * (imagenet_std[:, None, None].to(img6.device)) + (imagenet_mean[:, None, None].to(img6.device))
        img6 = img6.detach().permute(1, 2, 0).cpu().numpy()
        img6 = np.clip(img6, 0.0, 1.0)

    if data_07 is not None:
        img7 = data_07['imagec_1'][b_id]
        img7 = img7 * (imagenet_std[:, None, None].to(img7.device)) + (imagenet_mean[:, None, None].to(img7.device))
        img7 = img7.detach().permute(1, 2, 0).cpu().numpy()
        img7 = np.clip(img7, 0.0, 1.0)

    if data_08 is not None:
        img8 = data_08['imagec_1'][b_id]
        img8 = img8 * (imagenet_std[:, None, None].to(img8.device)) + (imagenet_mean[:, None, None].to(img8.device))
        img8 = img8.detach().permute(1, 2, 0).cpu().numpy()
        img8 = np.clip(img8, 0.0, 1.0)

    if data_09 is not None:
        img9 = data_09['imagec_1'][b_id]
        img9 = img9 * (imagenet_std[:, None, None].to(img9.device)) + (imagenet_mean[:, None, None].to(img9.device))
        img9 = img9.detach().permute(1, 2, 0).cpu().numpy()
        img9 = np.clip(img9, 0.0, 1.0)


    if is_origin_img:
        import torch.nn.functional as F
        from torchvision import transforms
        def _restore_img(img, mask):
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            img = img[F.interpolate(mask.unsqueeze(0).float(), mode='nearest', size=(832, 832))[0,0].bool().cpu().numpy().astype(bool)].reshape(new_wh[1], new_wh[0], 3)
            img = transforms.functional.resize(transforms.ToPILImage()(img), size=(origin_wh[1], origin_wh[0]), interpolation=transforms.InterpolationMode.BICUBIC)
            return np.array(img)


        img0 = _restore_img(img0, data_01['mask0'])
        img1 = _restore_img(img1, data_01['mask1'])

        if img2 is not None:
            img2 = _restore_img(img2, data_02['mask1'])
        
        if img3 is not None:
            img3 = _restore_img(img3, data_03['mask1'])

        if img4 is not None:
            img4 = _restore_img(img4, data_04['mask1'])
        if img5 is not None:
            img5 = _restore_img(img5, data_05['mask1'])
        if img6 is not None:
            img6 = _restore_img(img6, data_06['mask1'])
        if img7 is not None:
            img7 = _restore_img(img7, data_07['mask1'])
        if img8 is not None:
            img8 = _restore_img(img8, data_08['mask1'])
        if img9 is not None:
            img9 = _restore_img(img9, data_09['mask1'])


    num = len(data_01['mconf_f']) if len(data_01['mconf_f']) < topk else topk
    idx = torch.topk(data_01['mconf_f'], num, 0).indices
    mconf01 = data_01['mconf_f'][idx].cpu().numpy()
    kpts0_1 = data_01['mkpts0_f'][idx].detach().cpu().numpy()
    kpts1 = data_01['mkpts1_f'][idx].detach().cpu().numpy()

    num = len(data_02['mconf_f']) if len(data_02['mconf_f']) < topk else topk
    idx = torch.topk(data_02['mconf_f'], num, 0).indices
    mconf02 = data_02['mconf_f'][idx].cpu().numpy()
    kpts0_2 = data_02['mkpts0_f'][idx].detach().cpu().numpy()
    kpts2 = data_02['mkpts1_f'][idx].detach().cpu().numpy()

    num = len(data_03['mconf_f']) if len(data_03['mconf_f']) < topk else topk
    idx = torch.topk(data_03['mconf_f'], num, 0).indices
    mconf03 = data_03['mconf_f'][idx].cpu().numpy()
    kpts0_3 = data_03['mkpts0_f'][idx].detach().cpu().numpy()
    kpts3 = data_03['mkpts1_f'][idx].detach().cpu().numpy()

    num = len(data_04['mconf_f']) if len(data_04['mconf_f']) < topk else topk
    idx = torch.topk(data_04['mconf_f'], num, 0).indices
    mconf04 = data_04['mconf_f'][idx].cpu().numpy()
    kpts0_4 = data_04['mkpts0_f'][idx].detach().cpu().numpy()
    kpts4 = data_04['mkpts1_f'][idx].detach().cpu().numpy()

    num = len(data_05['mconf_f']) if len(data_05['mconf_f']) < topk else topk
    idx = torch.topk(data_05['mconf_f'], num, 0).indices
    mconf05 = data_05['mconf_f'][idx].cpu().numpy()
    kpts0_5 = data_05['mkpts0_f'][idx].detach().cpu().numpy()
    kpts5 = data_05['mkpts1_f'][idx].detach().cpu().numpy()

    num = len(data_06['mconf_f']) if len(data_06['mconf_f']) < topk else topk
    idx = torch.topk(data_06['mconf_f'], num, 0).indices
    mconf06 = data_06['mconf_f'][idx].cpu().numpy()
    kpts0_6 = data_06['mkpts0_f'][idx].detach().cpu().numpy()
    kpts6 = data_06['mkpts1_f'][idx].detach().cpu().numpy()

    num = len(data_07['mconf_f']) if len(data_07['mconf_f']) < topk else topk
    idx = torch.topk(data_07['mconf_f'], num, 0).indices
    mconf07 = data_07['mconf_f'][idx].cpu().numpy()
    kpts0_7 = data_07['mkpts0_f'][idx].detach().cpu().numpy()
    kpts7 = data_07['mkpts1_f'][idx].detach().cpu().numpy()

    num = len(data_08['mconf_f']) if len(data_08['mconf_f']) < topk else topk
    idx = torch.topk(data_08['mconf_f'], num, 0).indices
    mconf08 = data_08['mconf_f'][idx].cpu().numpy()
    kpts0_8 = data_08['mkpts0_f'][idx].detach().cpu().numpy()
    kpts8 = data_08['mkpts1_f'][idx].detach().cpu().numpy()

    num = len(data_09['mconf_f']) if len(data_09['mconf_f']) < topk else topk
    idx = torch.topk(data_09['mconf_f'], num, 0).indices
    mconf09 = data_09['mconf_f'][idx].cpu().numpy()
    kpts0_9 = data_09['mkpts0_f'][idx].detach().cpu().numpy()
    kpts9 = data_09['mkpts1_f'][idx].detach().cpu().numpy()


    if result_path:
        track_json = {}
        track_json['0to1'] = {}
        track_json['0to2'] = {}
        track_json['0to3'] = {}
        track_json['0to4'] = {}
        track_json['0to5'] = {}
        track_json['0to6'] = {}
        track_json['0to7'] = {}
        track_json['0to8'] = {}
        track_json['0to9'] = {}
        track_id = 0
        for kpt0, kpt1 in zip(kpts0_1, kpts1):

            track_json['0to1'][track_id] = {
                "start_id": 0,
                "end_id" : 1,
                "points": [kpt0.copy(), kpt1.copy()]}

            track_id += 1

        track_id = 0
        for kpt0, kpt2 in zip(kpts0_2, kpts2):

            track_json['0to2'][track_id] = {
                "start_id": 0,
                "end_id" : 2,
                "points": [kpt0.copy(), kpt2.copy()]}

            track_id += 1

        track_id = 0
        for kpt0, kpt3 in zip(kpts0_3, kpts3):

            track_json['0to3'][track_id] = {
                "start_id": 0,
                "end_id" : 3,
                "points": [kpt0.copy(), kpt3.copy()]}

            track_id += 1
        
        track_id = 0
        for kpt0, kpt4 in zip(kpts0_4, kpts4):

            track_json['0to4'][track_id] = {
                "start_id": 0,
                "end_id" : 4,
                "points": [kpt0.copy(), kpt4.copy()]}

            track_id += 1
        
        track_id = 0
        for kpt0, kpt5 in zip(kpts0_5, kpts5):

            track_json['0to5'][track_id] = {
                "start_id": 0,
                "end_id" : 5,
                "points": [kpt0.copy(), kpt5.copy()]}

            track_id += 1

        track_id = 0
        for kpt0, kpt6 in zip(kpts0_6, kpts6):

            track_json['0to6'][track_id] = {
                "start_id": 0,
                "end_id" : 6,
                "points": [kpt0.copy(), kpt6.copy()]}

            track_id += 1

        track_id = 0
        for kpt0, kpt7 in zip(kpts0_7, kpts7):

            track_json['0to7'][track_id] = {
                "start_id": 0,
                "end_id" : 7,
                "points": [kpt0.copy(), kpt7.copy()]}

            track_id += 1

        track_id = 0
        for kpt0, kpt8 in zip(kpts0_8, kpts8):

            track_json['0to8'][track_id] = {
                "start_id": 0,
                "end_id" : 8,
                "points": [kpt0.copy(), kpt8.copy()]}

            track_id += 1

        track_id = 0
        for kpt0, kpt9 in zip(kpts0_9, kpts9):

            track_json['0to9'][track_id] = {
                "start_id": 0,
                "end_id" : 9,
                "points": [kpt0.copy(), kpt9.copy()]}

            track_id += 1

    if is_origin_img:
        def _restore_kpts(kpts):
            kpts[:, 0] = kpts[: ,0] * origin_wh[0] / new_wh[0]
            kpts[:, 1] = kpts[: ,1] * origin_wh[1] / new_wh[1]
            return kpts

        for _0toN, tracks in track_json.items():
            for track_id, track in tracks.items():
                track['points'] = _restore_kpts(np.array(track['points'])).tolist()


    if result_path:
        import json
        for _0toN, tracks in track_json.items():
            for track_id, track in tracks.items():
                track['points'] = np.array(track['points']).tolist()
        with open(result_path, 'w') as f:
            json.dump(track_json, f, indent=2)


def make_matching_figures(data, mode='evaluation', path=None, dpi=150):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['confidence', 'evaluation', 'wheel']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['imagec_0'].size(0)):
        if mode == 'confidence':
            fig = make_confidence_figure(data, b_id, dpi=dpi, path=path)
        elif mode == 'evaluation':
            fig = make_evaluation_figure_color(data, b_id, dpi=dpi, path=path)
        elif mode == 'wheel':
            fig = make_evaluation_figure_wheel(data, b_id, path=path)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)
