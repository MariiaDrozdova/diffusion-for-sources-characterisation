#!/bin/bash

# Set the folder path containing the configs
config_folder="configs/pwv"

# Set the output folder prefix
output_folder_prefix="results/repeat_real_data_250"

# Loop through the configs in the folder
for config_file in ${config_folder}/*.yaml; do
    # Get the config file name without the extension
    config_name=$(basename "${config_file}" .yaml)

    # Deduce the output folder path
    output_folder="${output_folder_prefix}_${config_name}"

    # Build the Python command with the config file and run it
    python generate_images_realizations.py --ckpt "/home/drozdova/projects/ska-project/diffusion-true-ska/lightning_logs/2023-05-24-00-30-58_noisy_from_uv/version_0/checkpoints/best_epoch=246_step=384503.ckpt" --output "${output_folder}" --bs 50 --timestep_respacing 250 --config "${config_file}"
done

