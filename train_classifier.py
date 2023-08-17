from datetime import datetime
from pathlib import Path
import yaml
import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.trainer import ClassifierModule
from src.utils import get_config


def main(args) -> None:
    config_path = args.config
    auto_bs = args.auto_bs
    auto_lr = args.auto_lr
    config = get_config(config_path)

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    logger = pl_loggers.TensorBoardLogger(save_dir='lightning_logs', name=f"{date_time}_{config['comment']}")

    precision = 16 if config['fp16'] else 32
    accumulate_grad_batches = 1 if not config['accumulate_grad_batches'] else config['accumulate_grad_batches']
    iterations = config['iterations']
    eval_every = config['eval_every']

    module = ClassifierModule(config, config['fp16'])
    callback_lr = LearningRateMonitor('step')
    callback_best_ckpt = ModelCheckpoint(every_n_epochs=1, filename='best_{epoch}_{step}', monitor='val/loss',
                                         mode='min')
    callback_last_ckpt = ModelCheckpoint(every_n_epochs=1, filename='last_{epoch}_{step}')

    trainer = pl.Trainer(logger=logger,
                         callbacks=[
                             callback_lr,
                             callback_best_ckpt,
                             callback_last_ckpt
                         ],
                         gpus=-1, auto_select_gpus=True,
                         auto_scale_batch_size=auto_bs,
                         max_steps=iterations,
                         check_val_every_n_epoch=eval_every,
                         strategy='ddp',
                         precision=precision,
                         accumulate_grad_batches=accumulate_grad_batches)

    if auto_lr:
        lr_finder = trainer.tuner.lr_find(module, min_lr=1e-5, max_lr=1e-1,)
        lr = lr_finder.suggestion['lr']
        print(f"Suggested learning rate: {lr}")
        module.hparams.lr = lr
        config['lr_suggested'] = lr

    trainer.tune(module)

    # checkpoint
    path_checkpoint = config['fine_tune_from']
    trainer.fit(module, ckpt_path=path_checkpoint)

    # save config to file
    save_path = Path(logger.experiment.get_logdir()) / Path(config_path).name
    with open(save_path, 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/classifier.yaml', help='path to config file')
    parser.add_argument('--auto_bs', action='store_true', help='auto select batch size')
    parser.add_argument('--auto_lr', action='store_true', help='auto select learning rate')
    args = parser.parse_args()
    main(args)
