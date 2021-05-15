from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from src.pl_modules import get_pl_modules


def argparse():
    parser = ArgumentParser()
    parser.add_argument('cfg', type=str)
    parser.add_argument('--log_dir', type=str, default='./work_dirs')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser = pl.Trainer.add_argparse_args(parser)
    args, _ = parser.parse_known_args()
    return args


def main(args):
    cfg = OmegaConf.load(args.cfg)
    pl.seed_everything(cfg.seed)

    # configのtrain_paramsをargsに反映
    for k, v in cfg.train_params.items():
        setattr(args, k, v)

    # Disable default checkpoint callback
    args.checkpoint_callback = False
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = pl_loggers.TensorBoardLogger(save_dir=args.log_dir, name=Path(args.cfg).stem, default_hp_metric=False)
    trainer.callbacks.append(ModelCheckpoint(filename='{step:07d}-{val_loss:.2f}', monitor='val_loss',
                                             save_top_k=1, save_last=True))

    model, data = get_pl_modules(cfg)
    if args.ckpt_path:
        model = model.load_from_checkpoint(args.ckpt_path)
    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    args = argparse()
    main(args)
