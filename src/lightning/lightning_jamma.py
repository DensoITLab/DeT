from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path
import copy
from PIL import Image, ImageDraw
import tempfile

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from src.jamma.jamma import JamMa
from src.jamma.backbone import CovNextV2_nano
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics_test,
)
from src.utils.comm import gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from thop import profile
from src.utils.plotting import make_matching_figures
from src.utils.dataset import read_megadepth_depth, read_megadepth_color

class MatcherWrapper(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher

    def forward(self, data):
        # lightning の self.matcher(data, mode='test') が
        # (result, flops, runtime) みたいなタプルを返すなら、
        # 先頭だけ返しておけば OK（thop は中身にはあまり関心がない）
        out = self.matcher(data, mode='test')
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


class PL_JamMa(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.JAMMA_cfg = lower_config(_config['jamma'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)
        self.viz_path = Path('visualization')
        self.viz_path.mkdir(parents=True, exist_ok=True)
        # Matcher: JamMa
        self.backbone = CovNextV2_nano()
        self.matcher = JamMa(config=_config['jamma'], profiler=profiler)

        if pretrained_ckpt == 'official':
            state_dict = torch.hub.load_state_dict_from_url(
                'https://github.com/leoluxxx/JamMa/releases/download/v0.1/jamma.ckpt',
                file_name='jamma.ckpt')['state_dict']
            self.load_state_dict(state_dict, strict=True)
            logger.info(f"Load Official JamMa Weight")
        elif pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.load_state_dict(state_dict, strict=True)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        if _config['jamma']['use_compile']:
            self.matcher = torch.compile(
            self.matcher,
            mode="default",
            fullgraph=False,
            dynamic=False,
        )
            

        # Testing
        self.dump_dir = dump_dir
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0
        self.total_flops = 0
        self._flops_backbone = None
        self._flops_matcher = None
        self.warmup = False
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('number of params:', n_parameters / 1e6)

    def test_step(self, batch, batch_idx):
        with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
            det_eval = True
            device = 'cuda'
            if det_eval:
                print('pre_inference')
                img0_path = "data/megadepth/" + batch['pair_names'][0][0]
                with Image.open(img0_path) as im:
                    w, h = im.size

                    # 画像をコピーして編集
                    new_im = im.copy()
                    draw = ImageDraw.Draw(new_im)

                    margin = 8  # 黒枠の幅

                    # 上
                    draw.rectangle([0, 0, w, margin], fill=(0, 0, 0))
                    # 下
                    draw.rectangle([0, h - margin, w, h], fill=(0, 0, 0))
                    # 左
                    draw.rectangle([0, 0, margin, h], fill=(0, 0, 0))
                    # 右
                    draw.rectangle([w - margin, 0, w, h], fill=(0, 0, 0))

                    # 一時ファイルに保存
                    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    new_im.save(tmp.name)

                tent_imagec_0, tent_scale0, tent_mask0, tent_prepad_size0,_,_ = read_megadepth_color(tmp.name, 832, 8, padding=True)
                tent_mask0 = F.interpolate(tent_mask0[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
                tent_imagec_1 = batch['imagec_0']
                tent_mask1 = batch['mask0']
                tent_scale1 = batch['scale0']
                tent_prepad_size1 = batch['prepad_size0']

                first_batch = copy.deepcopy(batch)

                first_batch['imagec_0'] = tent_imagec_0.to(device)
                first_batch['imagec_1'] = tent_imagec_1.to(device)
                first_batch['scale0'] = tent_scale0.unsqueeze(0).to(device)
                first_batch['scale1'] = tent_scale1.to(device)
                first_batch['prepad_size0'] = tent_prepad_size0.unsqueeze(0).to(device)
                first_batch['prepad_size1'] = tent_prepad_size1.to(device)
                first_batch['mask0'] = tent_mask0.to(device)
                first_batch['mask1'] = tent_mask1.to(device)
                first_batch['custom_fine_thr'] = 0.1
                self.backbone(first_batch)
                self.matcher(first_batch)

                batch['prev_data'] = first_batch
                batch['algo_res'] = True
                batch['custom_fine_thr'] = 0.1
                batch['custom_fine_flex_thr'] = 0.1

            print('real_inference')
            self.start_event.record()
            with self.profiler.profile("Backbone"):
                flops1, _ = profile(self.backbone, inputs=(batch,), verbose=False)
                #self.backbone(batch)

            with self.profiler.profile("Matcher"):
                #self.matcher(batch, mode='test')
                flops2, _ = profile(self.matcher, inputs=(batch,), verbose=False)
            self.end_event.record()
            total_flops = flops1 + flops2
            self.total_flops += total_flops
            torch.cuda.synchronize()
            self.total_ms += self.start_event.elapsed_time(self.end_event)
            batch['runtime'] = self.start_event.elapsed_time(self.end_event)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        # Visualization #
        # path = str(self.viz_path) + '/' + str(batch_idx)
        # make_matching_figures(batch, 'confidence', path=path+'_confidence.png')
        # make_matching_figures(batch, 'evaluation', path=path+'_evaluation.png')
        # make_matching_figures(batch, 'wheel', path=path+'_wheel.png')

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                bs = batch['imagec_0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    epi_errs = batch['epi_errs'][mask].cpu().numpy()
                    correct_mask = epi_errs < 1e-4
                    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
                    n_correct = np.sum(correct_mask)
                    item['precision'] = precision
                    item['n_correct'] = n_correct
                    item['runtime'] = batch['runtime']
                    for key in ['R_errs', 't_errs']:
                        item[key] = batch[key][b_id][0]
                    dumps.append(item)
                ret_dict['dumps'] = dumps
        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics_test(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            print('Averaged Matching time over 1500 pairs: {:.2f} ms'.format(self.total_ms / 1500))
            print('Averaged FLOPs per pair: {:.2f} GMac'.format(self.total_flops / 1500 / 1e9))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'JAMMA_pred_eval', dumps)

    # def test_step(self, batch, batch_idx):
    #     flops1, params1 = profile(self.backbone, inputs=[batch])
    #     flops2, params2 = profile(self.matcher, inputs=[batch])
    #     return flops1+flops2
    #
    # def test_epoch_end(self, outputs):
    #     flops_mean = sum(outputs)/len(outputs) / 1e9
    #     print("mean flops: {}G".format(flops_mean))

        
    def _calc_flops_once(self, data):
        """入力サイズが同じ前提なら1回だけ FLOPs を計算してキャッシュ"""

        self.backbone.eval()
        # matcher も eval にしておく（必要なら）
        if hasattr(self.matcher, "eval"):
            self.matcher.eval()

        with torch.no_grad():
            # backbone の FLOPs
            flops_b, _ = profile(self.backbone, inputs=(data,), verbose=False)

            # matcher 用のラッパーモジュール
            wrapped = MatcherWrapper(self.matcher)
            wrapped.eval()  # 一応

            flops_m, _ = profile(wrapped, inputs=(data,), verbose=False)

        self._flops_backbone = flops_b
        self._flops_matcher = flops_m
        print(f"[FLOPs] backbone: {flops_b:,}, matcher: {flops_m:,}")


    def forward(self, data):
        self._calc_flops_once(data)
        if not self.warmup:
            print("Warming up...")
            # warm-up
            for _ in range(30):
                self.backbone(data)
                self.matcher(data, mode='test')
            self.warmup = True
            print("Warm-up done.")
        self.start_event.record()
        with self.profiler.profile("Backbone"):
            #flops1, _ = profile(self.backbone, inputs=(data,), verbose=False)
            self.backbone(data)
                

        with self.profiler.profile("Matcher"):
            #flops2, _ = profile(self.matcher, inputs=(data,), verbose=False)
            self.matcher(data, mode='test')
        self.end_event.record()
        #total_flops = flops1 + flops2
        total_flops = self._flops_backbone + self._flops_matcher
        torch.cuda.synchronize()
        run_time = self.start_event.elapsed_time(self.end_event)
        return data, total_flops, run_time
