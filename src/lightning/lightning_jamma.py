from collections import defaultdict
import pprint
import time
from loguru import logger
from pathlib import Path

import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from src.jamma.jamma import JamMa
from src.jamma.backbone import CovNextV2_nano
from src.jamma.utils.supervision import compute_supervision_fine, compute_supervision_coarse
from src.losses.loss import Loss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_f1,
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics_train_val, aggregate_metrics_test
)
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from thop import profile


class MatcherWrapper(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher

    def forward(self, data):
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
        # Matcher: JamMa
        self.backbone = CovNextV2_nano()
        self.matcher = JamMa(config=_config['jamma'], profiler=profiler)
        self.loss = Loss(_config)

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
        self.start_event = None
        self.end_event = None
        self.total_ms = 0
        self.total_flops = 0
        self._flops_backbone = None
        self._flops_matcher = None
        self.warmup = False
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {n_parameters / 1e6:.2f}M")

    def configure_optimizers(self):
        # Lightning 1.3 has limited scheduler recovery when resuming checkpoints.
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                     (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                     abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _train_inference(self, batch):

        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("Backbone"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                self.backbone(batch)

        with self.profiler.profile("Matcher"):
            self.matcher(batch, mode='train')

        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                self.loss(batch)

    def _val_inference(self, batch):

        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("Backbone"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                self.backbone(batch)

        with self.profiler.profile("Matcher"):
            self.matcher(batch, mode='val')

        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                self.loss(batch)

    def _compute_metrics_val(self, batch):
        with self.profiler.profile("Compute metrics"):
            compute_f1(batch)  # compute fi-score
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['imagec_0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'precision': batch['precision'],
                'recall': batch['recall'],
                'f1_score': batch['f1_score'],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def _compute_metrics(self, batch):
        with self.profiler.profile("Compute metrics"):
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['imagec_0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):

        self._train_inference(batch)
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            self.logger.experiment.add_scalar(f'train_loss', batch['loss'], self.global_step)

        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        self._val_inference(batch)
        ret_dict, _ = self._compute_metrics_val(batch)
        ret_dict['metrics'] = {**ret_dict['metrics'], 'max_matches': [batch['num_candidates_max']]}
        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}

        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }

    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics_train_val(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])

            # 3. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this
        logger.info(pprint.pformat(val_metrics_4tb))

    def _run_timed_inference(self, batch):
        use_cuda_timer = batch['imagec_0'].is_cuda and torch.cuda.is_available()
        if use_cuda_timer:
            if self.start_event is None or self.end_event is None:
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            start = time.perf_counter()

        with self.profiler.profile("Backbone"):
            self.backbone(batch)

        with self.profiler.profile("Matcher"):
            self.matcher(batch, mode='test')

        if use_cuda_timer:
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event)

        return (time.perf_counter() - start) * 1000.0

    def test_step(self, batch, batch_idx):
        device_type = 'cuda' if batch['imagec_0'].is_cuda else 'cpu'
        with torch.autocast(enabled=self.config.JAMMA.MP, device_type=device_type):
            if self._flops_backbone is None or self._flops_matcher is None:
                self._calc_flops_once(batch)
            total_flops = self._flops_backbone + self._flops_matcher
            self.total_flops += total_flops
            batch['runtime'] = self._run_timed_inference(batch)
            self.total_ms += batch['runtime']

        ret_dict, _ = self._compute_metrics(batch)

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
            logger.info(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics_test(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            num_pairs = max(len(metrics['identifiers']), 1)
            logger.info(
                'Averaged matching time over {} pairs: {:.2f} ms'.format(
                    num_pairs,
                    self.total_ms / num_pairs,
                )
            )
            logger.info(
                'Averaged FLOPs per pair: {:.2f} GMac'.format(
                    self.total_flops / num_pairs / 1e9
                )
            )
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'JAMMA_pred_eval', dumps)

    def _calc_flops_once(self, data):
        self.backbone.eval()
        if hasattr(self.matcher, "eval"):
            self.matcher.eval()

        with torch.no_grad():
            flops_b, _ = profile(self.backbone, inputs=(data,), verbose=False)

            wrapped = MatcherWrapper(self.matcher)
            wrapped.eval()

            flops_m, _ = profile(wrapped, inputs=(data,), verbose=False)

        self._flops_backbone = flops_b
        self._flops_matcher = flops_m
        logger.info(f"[FLOPs] backbone: {flops_b:,}, matcher: {flops_m:,}")

    def forward(self, data):
        if self._flops_backbone is None or self._flops_matcher is None:
            self._calc_flops_once(data)
        if not self.warmup:
            logger.info("Warming up...")
            for _ in range(30):
                self.backbone(data)
                self.matcher(data, mode='test')
            self.warmup = True
            logger.info("Warm-up done.")
        total_flops = self._flops_backbone + self._flops_matcher
        run_time = self._run_timed_inference(data)
        return data, total_flops, run_time
