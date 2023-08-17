# Radio-astronomical Image Reconstruction with Conditional Denoising Diffusion Model

Conditional DDPM for characterizing radio sources from dirty images.

This repository contains code for the corresponding paper.

## Training Instructions

1. Put the path to your data folder in `config/generator.yaml` (or to `toy_data` which is in the main folder).
2. Run the training script with the following command:

```
python train_generator.yaml --config config/generator.yaml
```
