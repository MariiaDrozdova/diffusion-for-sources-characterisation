"""Script that generate samples with classifier guidance"""

import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F

from src.trainer import GeneratorModule, ClassifierModule
from src.utils import get_config


def torch_to_image_numpy(tensor: torch.Tensor):
    tensor = tensor * 0.5 + 0.5
    im_np = [tensor[i].cpu().numpy().transpose(1, 2, 0)for i in range(tensor.shape[0])]
    return im_np


def main(args):
    config_gen = get_config(args.config_gen)
    config_clas = get_config(args.config_clas)

    config_gen['batch_size'] = args.batch_size

    path_ckpt_gen = args.ckpt_gen
    path_ckpt_clas = args.ckpt_clas

    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)

    module_gen = GeneratorModule.load_from_checkpoint(checkpoint_path=path_ckpt_gen,
                                                      config=config_gen, use_fp16=config_gen['fp16'],
                                                      timestep_respacing=str(args.timestep_respacing))
    module_gen.eval()
    module_clas = ClassifierModule.load_from_checkpoint(checkpoint_path=path_ckpt_clas,
                                                        config=config_clas, use_fp16=config_clas['fp16'])
    module_clas.eval()

    def cond_fn(x: torch.Tensor,
                t: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = module_clas(x_in, t)
            loss = F.mse_loss(logits, y)
            return torch.autograd.grad(loss, x_in)[0] * args.classifier_scale

    images = []
    generated_images = []
    labels = []

    for im_in, lbl in tqdm(module_gen.test_dataloader()):
        with torch.no_grad():
            im_out = module_gen.forward(lbl, progress=True, cond_fn=cond_fn)

        im_in_np = torch_to_image_numpy(im_in)
        im_out_np = torch_to_image_numpy(im_out)

        images.extend(im_in_np)
        generated_images.extend(im_out_np)
        labels.extend(lbl.cpu().numpy())

    images = np.array(images)
    generated_images = np.array(generated_images)
    labels = np.array(labels)

    path_images = output_path / 'images.npy'
    np.save(path_images, images)

    path_generated = output_path / 'generated_images.npy'
    np.save(path_generated, generated_images)

    path_labels = output_path / 'labels.npy'
    np.save(path_labels, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_gen', type=str, default='./configs/generator.yaml',
                        help='Path to generator config')
    parser.add_argument('--config_clas', type=str, default='./configs/classifier.yaml',
                        help='Path to classifier config')
    parser.add_argument('--ckpt_gen', type=str, required=True,
                        help='Path to generator checkpoint')
    parser.add_argument('--ckpt_clas', type=str, required=True,
                        help='Path to classifier checkpoint')
    parser.add_argument('--classifier_scale', type=float, default=3.,
                        help='Classifier scale for guidance. '
                             'Bigger - closes to label, less diversity')
    parser.add_argument('--batch_size', '--bs', '-b', type=int,
                        default=2,
                        help='Batch size to use when generating samples')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output folder')
    parser.add_argument('--timestep_respacing', '-t',
                        type=int, default=250)
    args = parser.parse_args()
    main(args)
