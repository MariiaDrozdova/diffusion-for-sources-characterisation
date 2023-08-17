from typing import Dict

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.data import MakeDataLoader
from src.model import create_classifier_and_diffusion, classifier_and_diffusion_defaults, create_named_schedule_sampler
from src.model import GalaxyZooClassifier


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = True


class ClassifierModule(pl.LightningModule):

    def __init__(self, config: Dict, use_fp16: bool = False):
        super().__init__()
        self.config = config

        # load data
        path_image = config['dataset']['image_path']
        path_labels = config['dataset']['label_path']
        size_image = config['dataset']['size']

        self.make_dl = MakeDataLoader(folder_images=path_image, file_labels=path_labels,
                                      image_size=size_image)

        # training parameters
        self.hparams.batch_size = config['batch_size']  # set start batch_size
        self.hparams.lr = eval(config['lr'])

        # load encoder, diffusion model and classifier
        params = classifier_and_diffusion_defaults()
        params['classifier_use_fp16'] = use_fp16
        self.encoder, self.diffusion = create_classifier_and_diffusion(**params)
        if use_fp16:
            self.encoder.convert_to_fp16()

        n_in = self.encoder.out_channels
        n_out = config['dataset']['n_classes']
        self.classifier = GalaxyZooClassifier(n_in, n_out)

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
        opt = optim.AdamW(list(self.encoder.parameters()) + list(self.classifier.parameters()),
                          lr=lr, weight_decay=wd)
        return [opt]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def step(self, batch, batch_idx: int, *, stage: str) -> torch.Tensor:
        im, label = batch

        t, _ = self.schedule_sampler.sample(im.shape[0], device=self.device)
        im = self.diffusion.q_sample(im, t)

        logits = self.classifier(self.encoder(im, t))
        loss = F.mse_loss(logits, label)
        self.log(f'{stage}/loss', loss)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='val')

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x, t))
