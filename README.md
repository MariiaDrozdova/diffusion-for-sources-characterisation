# Radio-astronomical Image Reconstruction with Conditional Denoising Diffusion Model

This repository contains the implementation of the Conditional Denoising Diffusion Probabilistic Model (DDPM) for characterizing radio sources from dirty images, as detailed in our corresponding paper.

## Data Preparation

The code requires a specific dataset format. An example of this format can be found in `example_data/toy_data`. Each image should be saved as a separate .npy file. Dirty images are stored in the `dirty` folder, while sky model images are in the `true` folder.

For real data testing (datasets without ground truth), the dataset folder name should include `real_data`. An example can be found at `example_data/toy_real_data`.

To create a dataset in the required format, refer to the provided notebook.

We've shared thirty files from our datasets in `example_data/mock_data` and `example_data/mock_real_data`. To access the complete dataset, please contact Omkar Beit.

## Training Instructions

1. Set the path to your data folder in `config/generator.yaml` (or use the provided `toy_data` folder) under the `dataset.image_path` section.
2. Initiate the training using the command:
```
python train_generator.yaml --config config/generator.yaml
```

## Testing Instructions 

### Reconstructing Sky Models
To reconstruct sky models from dirty images, use the command:

```
python generate_images_realizations.py --ckpt /home/drozdova/projects/diffusion-for-sources-characterisation/lightning_logs/2023-08-17-13-14-37_toy_run/version_0/checkpoints/best_epoch=2_step=54.ckpt --output results/experiment1 --bs 1 --timestep_respacing 250 --runs_per_sample 5
```
This command will create a folder `results/experiment1_power2`, saving each batch in separate files with the specified number of runs per sample. While our paper used 20 runs per sample, 5 seems to suffice for most cases. The "power2" suffix is added automatically from the config file for further processing.

### Sources Characterization

There are two provided scripts: one for datasets with ground truth and another for datasets without.

- **With Ground Truth Data**:
```
python make_catalog_with_gt.py --runs_per_sample 5 --folders results/mock_experiment1_power2/
```
- **Without Ground Truth Data**:

If you have images of the sky that are consistent, set the `consistent_ra_dec` flag to True. For toy datasets, this isn't applicable as sources are placed randomly in each image with randomized ra and dec values.


```
python make_catalog_with_real_data.py --runs_per_sample 5 --consistent_ra_dec True --folders results/mock_experiment1_power2/
```

For the aggregation detect-aggregate method (detailed in our paper), when `consistent_ra_dec` is set to True, sources will be merged across all provided images. The output can be found in the corresponding folder with the filename `aggregated_sources.csv` (e.g., "results/mock_experiment1_power2/aggregated_sources.csv").

## Contact

For any questions regarding dataset used in the paper please contact Omkar Beit at [omkar.bait@unige.ch](mailto:omkar.bait@unige.ch).
For any questions regarding paper and code please contact Mariia Drozdova at [mariia.drozdova@unige.ch](mailto:mariia.drozdova@unige.ch).

