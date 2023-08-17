# Radio-astronomical Image Reconstruction with Conditional Denoising Diffusion Model

Conditional DDPM for characterizing radio sources from dirty images.

This repository contains code for the corresponding paper.

## Data Preparation

The code is suited for very specific dataset format. The example is shown in `toy_data`. Each image is saved in a separate .npy file. Dirty images in the folder `dirty` and sky model images in the folder `true`.

## Training Instructions

1. Put the path to your data folder in `config/generator.yaml` (or to `toy_data` which is in the main folder).
2. Run the training script with the following command:
```
python train_generator.yaml --config config/generator.yaml
```
## Testing Instructions 

1. For reconstructing sky models from dirty images:
```
python generate_images_realizations.py --ckpt /home/drozdova/projects/diffusion-for-sources-characterisation/lightning_logs/2023-08-17-13-14-37_toy_run/version_0/checkpoints/best_epoch=2_step=54.ckpt --output results/experiment1 --bs 1 --timestep_respacing 250 --runs_per_sample 5
```
This will create a folder `results/experiment1_power2` where it will write each batch in a separate file with the chosen number of runs per sample. (In the paper we used 20, but 5 seems to be good enough) 
The adding of "power2" will be done automatically. The power is written from config file and is needed for further processing.

2. For sources characterisation
