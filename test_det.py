import argparse
import pprint
import torch
import torch.nn as nn
import numpy as np

import pytorch_lightning as pl
from loguru import logger as loguru_logger

from src.config.default import get_cfg_defaults
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_jamma import PL_JamMa
from src.utils.profiler import build_profiler


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_cfg_path', type=str, default="configs/data/megadepth_test_1500.py")
    parser.add_argument('--main_cfg_path', type=str, default="configs/jamma/outdoor/test.py")
    parser.add_argument('--ckpt_path', type=str, default="official")
    parser.add_argument('--dump_dir', type=str, default="dump/jamma_outdoor")
    parser.add_argument('--profiler_name', type=str, default="inference")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--thr', type=float, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    args.gpus = 1
    args.accelerator = 'gpu'
    args.benchmark = True
    print(args)
    pprint.pprint(vars(args))

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
    loguru_logger.info("JamMa-lightning initialized!")

    # data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info("DataModule initialized!")
    data_module.setup(stage="test")
    loader = data_module.test_dataloader()

    # =====（任意）通常の評価 =====
    trainer = pl.Trainer.from_argparse_args(args, replace_sampler_ddp=False, logger=False)
    loguru_logger.info("Start testing!")
    trainer.test(model, datamodule=data_module, verbose=False)

