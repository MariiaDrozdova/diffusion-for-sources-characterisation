from typing import Dict
import copy

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.model import model_and_diffusion_defaults, create_model_and_diffusion
from src.model import create_named_schedule_sampler
from src.model.resample import LossAwareSampler
from src.data import MakeDataLoader
from src.utils import update_ema


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = True


class GeneratorModule(pl.LightningModule):

    def __init__(self, config: Dict, use_fp16: bool = False,
                 timestep_respacing=None,):
        super().__init__()
        self.config = config

        # load data
        path_image = config['dataset']['image_path']
        path_labels = config['dataset']['label_path']
        real_data = config['dataset']['real_data']
        self.size_image = config['dataset']['size']
        self.n_channels = config['dataset']['n_channels']
        self.clip_denoised = config['clip_denoised']

        use_zeros = config['dataset']['use_zeros']
        from_uv = config['dataset']['from_uv']
        power = config['dataset']['power']


        self.make_dl = MakeDataLoader(folder_images=path_image,
                                      image_size=self.size_image, real_data=real_data,
                                      power=power, use_zeros=use_zeros, from_uv=from_uv
                                      )

        # training parameters
        self.hparams.batch_size = config['batch_size']  # set start batch_size
        self.hparams.lr = eval(config['lr'])

        # load model and diffusion
        params = model_and_diffusion_defaults()
        params['image_size'] = self.size_image
        params['n_classes'] = config['dataset']['n_classes']
        params['use_fp16'] = use_fp16
        params['in_channels'] = 2 * config['dataset']['n_channels']
        params['out_channels'] = config['dataset']['n_channels']
        params['use_y_conditioning'] = config['model']['use_y_conditioning']
        params['timestep_respacing'] = timestep_respacing
        params['diffusion_steps'] = config['model']['diffusion_steps']

        self.model, self.diffusion = create_model_and_diffusion(**params)

        self.ema_rate = config['ema_rate']
        self.model_ema = copy.deepcopy(self.model).eval()

        if use_fp16:
            self.model.convert_to_fp16()

        # load sampler
        self.schedule_sampler = create_named_schedule_sampler(
            config['schedule_sampler'], self.diffusion
        )

    def get_dataloader(self, mode: str) -> DataLoader:
        """Returns dataloader

        :param mode: type of dataloader to return. Choices: train, val, test
        :return: dataloader
        """

        bs = self.hparams.batch_size
        n_workers = self.config['n_workers']

        if mode == 'train':
            return self.make_dl.get_data_loader_train(batch_size=bs, shuffle=True, num_workers=n_workers)
        elif mode == 'val':
            return self.make_dl.get_data_loader_valid(batch_size=bs, shuffle=False, num_workers=n_workers)
        elif mode == 'test':
            return self.make_dl.get_data_loader_test(batch_size=bs, shuffle=False, num_workers=n_workers)
        else:
            raise ValueError('mode must be one of train, val, test')

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader('val')

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader('test')

    def configure_optimizers(self):
        lr = self.hparams.lr
        wd = eval(self.config['wd'])
        opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        return [opt]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def step(self, batch, batch_idx: int, *, stage: str) -> torch.Tensor:
        data = batch
        im = data["true"]
        dirty_noisy = data["dirty_noisy"]

        input = im

        t, weights = self.schedule_sampler.sample(input.shape[0], device=self.device)
        losses = self.diffusion.training_losses(
            self.model,
            input,
            t,
            cond=dirty_noisy,
            model_kwargs={},
        )

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                    t, losses['loss'].detach()
            )

        loss = (losses['loss'] * weights).mean()
        res_dict = {f'{stage}/weighted_loss': loss}
        for k, v in losses.items():
            res_dict[f'{stage}/{k}'] = v.mean()
        self.log_dict(res_dict)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='val')

    def on_after_backward(self) -> None:
        self._update_ema()

    def _update_ema(self) -> None:
        update_ema(self.model_ema.parameters(), self.model.parameters(), rate=self.ema_rate)

    def forward(self, y1: torch.Tensor,y2: torch.Tensor,  denoised_fn=None, cond_fn=None, progress: bool = False) -> torch.Tensor:
        b = y1.shape[0]
        shape = (b, self.n_channels, self.size_image, self.size_image)
        return self.diffusion.p_sample_loop(self.model_ema, shape,
                                            clip_denoised=self.clip_denoised,
                                            denoised_fn=denoised_fn,
                                            cond_fn=cond_fn,
                                            model_kwargs={'y1': y1, 'y2': y2},
                                            progress=progress)
