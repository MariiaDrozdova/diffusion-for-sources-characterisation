# Radio-astronomical Image Reconstruction with Conditional Denoising Diffusion Model

Conditional DDPM for characterizing radio sources from dirty images.

This repository contains code for the corresponding paper.

## Data Preparation

The code is suited for very specific dataset format. The example is shown in `example_data/toy_data`. Each image is saved in a separate .npy file. Dirty images in the folder `dirty` and sky model images in the folder `true`.

In case real data should be tested (the one without ground truth), it is also possible, the corresponding dataset folder should contain `real_data` in the name. The example is at `example_data/toy_real_data`.

You can generate dataset of this type yourself, using the code provided in the notebook.

We provide thirty files from our datasets in `example_data/mock_data` and `example_data/mock_real_data`. If you would liek to accesss to the whole dataset, please contact Omkar Beit.

## Training Instructions

1. Put the path to your data folder in `config/generator.yaml` (or to `toy_data` which is in the main folder) under `dataset.image_path` .
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

We provide two files for the case when there is and there is no ground truth data. 
Both will create multiple files in the specified folder.


```
python make_catalog_with_gt.py --runs_per_sample 5 --folders results/mock_experiment1_power2/
```

```
python make_catalog_with_real_data.py --runs_per_sample 5 --consistent_ra_dec True --folders results/mock_experiment1_power2/
```
consistent_ra_dec should be set to True if we have images of the sky which are consistent. In case of toy data it is not applicable: we randomly put sources in each image and randomly generate ra, dec. For real data we expect to see the source in all images who has its (ra, dec) within the view. For the aggregation detect-aggregate (see our paper for more details) when consistent_ra_dec = True the sources will be merged through all images provided. The output will be in the correspondign folder under the name aggregated_sources.csv. In our example it is "results/mock_experiment1_power2/aggregated_sources.csv"

