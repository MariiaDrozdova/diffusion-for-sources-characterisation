import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np

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

    shape = (args.batch_size * 2, module.n_channels, module.size_image, module.size_image)

    # generate images
    dls = [module.val_dataloader(), module.test_dataloader()]
    names = ['val', 'test']

    for i, dl in enumerate(dls):
        images = []
        generated_images = []
        labels1 = []
        labels2 = []
        dirty_noisy_list = []
        gt_sources_list = []

        #iter_loader = iter(dl)
        #batch = next(iter_loader)
        repeat_param = 20
        for batch_idx, batch in tqdm(enumerate(dl)):
            print(batch_idx)

            images = []
            generated_images = []
            labels1 = []
            labels2 = []
            dirty_noisy_list = []
            gt_sources_list = []

            im_in, lbl1, lbl2 = batch["true"], batch["magnitude"], batch["phase"]
            im_in_ = im_in.to(device)
            lbl1_ = lbl1.to(device)
            lbl2_ = lbl2.to(device)

            dirty_noisy = batch["dirty_noisy"].to(device)
            filenames = batch["filename"]

            for j in tqdm(range(repeat_param)):
                lbl1 = lbl1_
                lbl2 = lbl2_
                im_in = im_in_

                with torch.no_grad():
                    zero_label = torch.zeros_like(lbl1, device=device)
                    lbl1 = torch.cat([lbl1, zero_label], dim=0)
                    lbl2 = torch.cat([lbl2, zero_label], dim=0)

                    zero_label_noise = torch.zeros_like(dirty_noisy, device=device)
                    dirty_noisy = torch.cat([dirty_noisy, zero_label_noise], dim=0)
                    im_out = diffusion.p_sample_loop(
                        model_fn,
                        cond=dirty_noisy,
                        shape=shape,
                        device=device,
                        clip_denoised=True,
                        progress=False,
                        model_kwargs={'y1': lbl1, 'y2': lbl2,},
                        cond_fn=None,
                    )[:args.batch_size]
                    lbl1 = lbl1[:args.batch_size]
                    lbl2 = lbl2[:args.batch_size]
                    dirty_noisy = dirty_noisy[:args.batch_size]

                im_in = torch_to_image_numpy(im_in)
                im_out = torch_to_image_numpy(im_out)
                dirty_noisy_ = torch_to_image_numpy(dirty_noisy)
                lbl1 = lbl1.cpu().numpy()
                lbl2 = lbl2.cpu().numpy()

                images.extend(im_in)
                generated_images.extend(im_out)
                labels1.extend(lbl1)
                labels2.extend(lbl2)
                dirty_noisy_list.extend(dirty_noisy_)
                gt_sources_list.extend(filenames)


            images = np.array(images)
            generated_images = np.array(generated_images)
            generated_images = np.array(generated_images)
            labels1 = np.array(labels1)
            labels2 = np.array(labels2)

            np.save(output_path / f'batch={batch_idx}_{names[i]}_images.npy', images)
            np.save(output_path / f'batch={batch_idx}_{names[i]}_generated_images.npy', generated_images)
            np.save(output_path / f'batch={batch_idx}_{names[i]}_labels1.npy', labels1)
            np.save(output_path / f'batch={batch_idx}_{names[i]}_labels2.npy', labels2)
            np.save(output_path / f'batch={batch_idx}_{names[i]}_dirty_noisy.npy', dirty_noisy_list)
            np.save(output_path / f'batch={batch_idx}_{names[i]}_gt_sources.npy', gt_sources_list)


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
