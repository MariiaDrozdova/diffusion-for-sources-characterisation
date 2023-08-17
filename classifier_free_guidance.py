import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch

from src.trainer import GeneratorModule
from src.utils import get_config


def torch_to_image_numpy(tensor: torch.Tensor):
    tensor = tensor * 0.5 + 0.5
    im_np = [tensor[i].cpu().numpy().transpose(1, 2, 0) for i in range(tensor.shape[0])]
    return im_np


def main(args) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_path = args.config
    config = get_config(config_path)
    config['batch_size'] = args.batch_size

    path_checkpoint = args.ckpt
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)

    module = GeneratorModule.load_from_checkpoint(checkpoint_path=path_checkpoint,
                                                  config=config, use_fp16=config['fp16'],
                                                  timestep_respacing=str(args.timestep_respacing))
    module.eval()

    diffusion = module.diffusion
    model = module.model
    model = model.to(device)

    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + args.guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    images = []
    generated_images = []
    labels = []

    shape = (args.batch_size * 2, module.n_channels, module.size_image, module.size_image)
    for im_in, lbl in tqdm(module.test_dataloader()):
        im_in = im_in.to(device)
        lbl = lbl.to(device)

        with torch.no_grad():

            zero_label = torch.zeros_like(lbl, device=device)
            lbl = torch.cat([lbl, zero_label], dim=0)
            im_out = diffusion.p_sample_loop(
                model_fn,
                shape=shape,
                device=device,
                clip_denoised=True,
                progress=False,
                model_kwargs={'y': lbl},
                cond_fn=None,
            )[:args.batch_size]

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
    parser.add_argument('--config', '-c', type=str,
                        default='./configs/generator.yaml',
                        help='Path to config')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output folder')
    parser.add_argument('--batch_size', '--bs', '-b', type=int,
                        default=50,
                        help='Batch size to use when generating samples')
    parser.add_argument('--guidance_scale', '-s', type=float,
                        default=3.)
    parser.add_argument('--timestep_respacing', '-t',
                        type=int, default=250)
    args = parser.parse_args()
    main(args)
