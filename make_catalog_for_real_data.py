#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import re
import pickle
import warnings

import tqdm
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from functools import partial

from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy import signal
from sklearn.cluster import DBSCAN
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.wcs import FITSFixedWarning
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from photutils.segmentation import deblend_sources, SourceCatalog
from photutils import detect_sources
import astropy.units as u
from glob import glob
from IPython.display import display

from src.utils import get_config

warnings.simplefilter('ignore', category=FITSFixedWarning)

NOISE_PER_IMAGE = 5e-5 #in our experiment noise amplitude is constant

astropy_columns = [
    "ra",  # /xcentroid",
    "dec",  # /ycentroid",
    "source_sum",
    "area",
    "equivalent_radius",
    "semiminor_sigma",
    "semimajor_sigma",
    "max_value",
    "min_value",
    "eccentricity",
    "orientation",
]
casa_columns = [
    "ra",
    "dec",
    "flux",
    "SNR",
    "SNR normalized",
    "minor",
    "major",
]


def initialize_snr_metrics():
    return {
        "snrs": None,
        "purity": None,
        "completeness": None,
        "nb_points": None,
        "tp": 0,
        "fp": 0,
        "fn": 0
    }


def initialize_of_nbs_metrics():
    return {
        "purity": None,
        "completeness": None
    }


def initialize_final_metric_entry():
    return {
        "resonstruction_metrics": {},
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "purity": 0,
        "completeness": 0,
        "snr": initialize_snr_metrics(),
        "snrnorm": initialize_snr_metrics(),  # Assuming snrnorm has the same structure as snr
        "of_nbs": initialize_of_nbs_metrics(),
        "values": None
    }


def initialize_final_metrics(runs_per_sample):
    metrics = {
        "mean": initialize_final_metric_entry(),
        "medoid": initialize_final_metric_entry(),
        "median": initialize_final_metric_entry(),
    }

    for i in range(runs_per_sample):
        metrics[f"individual_{i}"] = initialize_final_metric_entry()
        metrics[f"localized_threshold_{i}"] = initialize_final_metric_entry()

    return metrics

# Function to cluster points (ra, dec) within each image using DBSCAN
def cluster_points(dataframe, image_idx, eps=5e-5):
    # Filter data for specific image
    if image_idx is not None:
        image_data = dataframe[dataframe["image_idx"] == image_idx]
    else:
        image_data = dataframe

    # Apply DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=2).fit(image_data[['ra', 'dec']])
    image_data = image_data.copy()
    image_data['cluster'] = clustering.labels_

    return image_data

def compute_psnr_ssim(image1, image2):

    # Convert images to grayscale if necessary
    if image1.ndim == 3:
        image1 = np.mean(image1, axis=2)
    if image2.ndim == 3:
        image2 = np.mean(image2, axis=2)

    # Compute PSNR
    dtype = image1.dtype
    image2 = image2.astype(dtype)

    psnr = peak_signal_noise_ratio(image1, image2)

    # Compute SSIM
    ssim = structural_similarity(image1, image2,data_range=1.0)

    # Return PSNR and SSIM values
    return psnr, ssim

def prepareHeaders(header):
    header.remove("HISTORY", remove_all=True)
    header.remove("OBJECT", remove_all=True)
    header.remove("DATE", remove_all=True)
    header.remove("DATE-OBS", remove_all=True)
    header.remove("TIMESYS", remove_all=True)
    header.remove("ORIGIN", remove_all=True)
    return header

def image_to_sources(
        image, verbose=False,
        image_size=512,
        detect_npixels=15
):
    if np.max(image) < 1e-10:
        return None  # no sources
    # Smooth the image with a Gaussian kernel to help with source detection
    if image_size == 128:
        sigma_clipped_stats_sigma = 3.0
        detect_npixels = 5
        deblend_sources_npixels = 1
        std_const = 5

    if image_size == 512:
        sigma_clipped_stats_sigma = 2  # 25.0#10.0
        detect_npixels = 10  # detect_npixels
        deblend_sources_npixels = 10
        std_const = 120  # 12

    # Calculate the threshold for source detection
    mean, median, std = sigma_clipped_stats(image, sigma=sigma_clipped_stats_sigma)
    threshold = (std_const * std)

    # Detect the sources in the image
    segm = detect_sources(image, threshold, npixels=detect_npixels)

    # Deblend the sources
    try:
        segm_deblend = deblend_sources(image, segm, npixels=deblend_sources_npixels,
                                       progress_bar=None)  # , deblend_cont=0.01)
    except ValueError:
        return None
    props = SourceCatalog(image, segm_deblend)

    coords = []

    # Print the properties of each detected source
    for prop in props:
        if verbose:
            print('Xcentroid =', prop.xcentroid.value)
            print('Ycentroid =', prop.ycentroid.value)
            print('Source area =', prop.area.value)
            print('Source integrated flux =', prop.source_sum)
        coords.append([
            prop.xcentroid,
            prop.ycentroid,
            prop.data.sum(),
            prop.area.value,
            prop.equivalent_radius.value,
            0.1 * 2.3548 * prop.semiminor_sigma.value,
            0.1 * 2.3548 * prop.semimajor_sigma.value,
            prop.max_value,
            prop.min_value,
            (prop.eccentricity * u.m).value,
            (prop.orientation * u.m).value,
        ])
    coords = np.array(coords)
    return coords


def true_trasnform(true, power):
    const = (0.7063881) ** (30.0 / power)
    true = (true) ** (1. / power)
    true = (true) / const
    true = (true - 0.5) / 0.5
    return true


def true_itrasnform(true, power):
    const = (0.7063881) ** (30.0 / power)
    true_back = true * const
    true_back = (true_back) ** (power)
    return true_back


def actro_to_pix(true_sources, wcs):
    true_sources = np.array(true_sources)
    coords_deg = np.hstack((true_sources[:, 0:1], true_sources[:, 1:2], np.zeros((true_sources.shape[0], 2))))
    coords_pix = wcs.wcs_world2pix(coords_deg, 0)
    gt_sources = np.array(coords_pix)[:, :4]  # *128/512
    gt_sources = gt_sources.astype(int).astype(float)
    return gt_sources


def pix_to_astro(pixel_coords, wcs):
    if pixel_coords is None:
        return pixel_coords
    pixel_coords = np.array(pixel_coords)
    coords_pix = np.hstack((pixel_coords[:, 0:1], pixel_coords[:, 1:2], np.zeros((pixel_coords.shape[0], 2))))
    coords_deg = wcs.wcs_pix2world(coords_pix, 0)
    celestial_coords = np.array(coords_deg)[:, :2]
    pixel_coords[:, :2] = celestial_coords
    astro_coords = pixel_coords
    return astro_coords


def im_reshape(downsampled_array):
    if downsampled_array.shape[1] == 512:
        return downsampled_array
    im_size = downsampled_array.shape[1]
    scale_factor = 512 // im_size
    downsampled_image = torch.tensor(downsampled_array).reshape(1, 1, im_size, im_size, )
    upsampled_image = torch.nn.functional.interpolate(downsampled_image, scale_factor=scale_factor, mode='bicubic')
    upsampled_image = upsampled_image.data.numpy()[0, 0, :, :, ]
    return upsampled_image


def add_column_i(sources, i):
    if sources is None:
        return None
    N = sources.shape[0]
    column_to_add = i * np.ones((N, 1))
    result = np.hstack((sources, column_to_add))
    return result


def compute_reconstruction_metrics_from_im(gen_im, im, verbose=False, power=None):
    l2_dif = np.sqrt(np.sum(np.square(gen_im - im)))
    l1_dif = np.sum(np.abs(gen_im - im))
    image1 = true_trasnform(gen_im, power) / 2 + 0.5
    #

    image2 = true_trasnform(im, power) / 2 + 0.5
    image1[image1 > 1] = 1
    image2[image2 > 1] = 1
    image1[image1 < 0] = 0
    image2[image2 < 0] = 0

    # image2 = image2*np.max(image1)/np.max(image2)

    if verbose:
        plt.imshow(image1)
        plt.show()
        plt.imshow(image2)
        plt.show()

    psnr, ssim = compute_psnr_ssim(image1, image2)

    if verbose:
        print(ssim, psnr)
    reconstruction_metrics = [
        l2_dif,
        l1_dif,
        psnr,
        ssim,
    ]
    return reconstruction_metrics


def compute_localization(current, filtering_for_fp=None, verbose=False):
    # Count the number of NaNs in each column where sources detected - false positive
    if filtering_for_fp is None:
        fp = current[["flux"]].isna().any(axis=1).sum()
    else:
        fp = current[["flux"]][filtering_for_fp].isna().any(axis=1).sum()

    # Count the number of NaNs in column diff_idx in results - false negative
    fn = current[["area_predicted"]].isna().any(axis=1).sum()
    tp = current[["flux", "area_predicted"]].notna().all(axis=1).sum()
    if verbose:
        print(fp, tp, fn)
        print("purity=", tp / (tp + fp))
        print("completeness=", tp / (tp + fn))
    return tp, fp, fn


def merge_pd_frames(gt_sources_, predicted_sources_, verbose=False):
    # Threshold for distance
    eps = 5e-5

    predicted_sources = predicted_sources_.copy()
    predicted_sources['index'] = predicted_sources.index

    gt_sources = gt_sources_.copy()
    gt_sources['index'] = gt_sources.index

    # Dataframe to store the result
    columns = list(gt_sources.columns) + [col + '_predicted' for col in predicted_sources.columns]
    result = pd.DataFrame(columns=columns)

    # Loop through each unique image_idx in gt_sources
    for image_idx in set(gt_sources['image_idx']):
        # Filter data for the current image_idx
        gt_filtered = gt_sources[gt_sources['image_idx'] == image_idx].reset_index(drop=True)
        pred_filtered = predicted_sources[predicted_sources['image_idx'] == image_idx].reset_index(drop=True)
        if verbose:
            print(image_idx)
            display(gt_filtered)
            display(pred_filtered)

        # Check if either gt_filtered or pred_filtered is empty
        if len(gt_filtered) == 0 or len(pred_filtered) == 0:
            # Append rows from gt_filtered or pred_filtered with no match
            for i in range(len(gt_filtered)):
                row = pd.concat([gt_filtered.iloc[i], pd.Series([np.nan] * len(predicted_sources.columns),
                                                                index=[col + '_predicted' for col in
                                                                       predicted_sources.columns])])
                result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
            for j in range(len(pred_filtered)):
                row = pd.concat([pd.Series([np.nan] * len(gt_sources.columns), index=gt_sources.columns),
                                 pred_filtered.iloc[j].add_suffix('_predicted')])
                result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
        else:
            # Compute squared differences for ra and dec
            ra_diff = (gt_filtered['ra'].values[:, np.newaxis] - pred_filtered['ra'].values) ** 2
            dec_diff = (gt_filtered['dec'].values[:, np.newaxis] - pred_filtered['dec'].values) ** 2

            # Compute squared distances
            squared_distances = ra_diff + dec_diff

            # Find the closest neighbor for each point in gt_filtered
            closest_indices = np.argmin(squared_distances, axis=1)
            closest_distances = squared_distances[np.arange(len(gt_filtered)), closest_indices]

            # Match points in gt_filtered to their closest neighbors in pred_filtered
            for i in range(len(gt_filtered)):
                # Check if the closest neighbor is within the threshold
                if closest_distances[i] < eps ** 2:
                    # Join the matched rows
                    row = pd.concat(
                        [gt_filtered.iloc[i], pred_filtered.iloc[closest_indices[i]].add_suffix('_predicted')])
                else:
                    # Join with NaN for missing values from predicted_sources
                    row = pd.concat([gt_filtered.iloc[i], pd.Series([np.nan] * len(predicted_sources.columns),
                                                                    index=[col + '_predicted' for col in
                                                                           predicted_sources.columns])])

                # Append the row to the result dataframe
                result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)

            # Mark the matched rows in pred_filtered as used
            pred_filtered_used = pred_filtered.iloc[closest_indices]
            pred_filtered_unused = pred_filtered[~pred_filtered.index.isin(pred_filtered_used.index)]
            if verbose:
                print("matched")
                display(pred_filtered_used)
                print("unmatched")
                display(pred_filtered_unused)

            # Append unused rows from pred_filtered with no match in gt_filtered
            if len(pred_filtered_unused) > 0:
                for _, row in pred_filtered_unused.iterrows():
                    row = pd.concat([pd.Series([np.nan] * len(gt_sources.columns), index=gt_sources.columns),
                                     row.add_suffix('_predicted')])
                    result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)

    # Append remaining rows from predicted_sources with no match in gt_sources
    indexes = predicted_sources.index.isin(result.index_predicted)
    unmatched_predicted = predicted_sources[~indexes]
    indexes = indexes + 0

    unmatched_nb = len(unmatched_predicted)

    if len(unmatched_predicted) > 0:
        for _, row in unmatched_predicted.iterrows():
            row = pd.concat(
                [pd.Series([np.nan] * len(gt_sources.columns), index=gt_sources.columns), row.add_suffix('_predicted')])
            result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
    else:
        pass

    return result, unmatched_nb


def plot_generated_images(
        true,
        generated,
        artificial_dirty_im,
        uncertainty=None,
        sources=(None, None, None),
        save_fig=False,
        save_name=None,
):
    clean_sources, generated_sources, true_sources = sources
    if uncertainty is not None and np.sum(uncertainty) == 0:
        if np.sum(true - generated) == 0:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
            # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
    else:
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 4))

    axes[0].set_title("noisy dirty image")
    im0 = axes[0].imshow(artificial_dirty_im)
    if true_sources is not None:
        true_sources = true_sources[(true_sources[:, 0] >= 0) & (true_sources[:, 0] <= 512) &
                                    (true_sources[:, 1] >= 0) & (true_sources[:, 1] <= 512)]
        axes[0].scatter(true_sources[:, 0], true_sources[:, 1], s=90, c='none', marker="o", edgecolor='r', linewidths=1)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    cbar = plt.colorbar(im0, ax=axes[0])
    # Set the colorbar tick format
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-4, 4))
    cbar.ax.yaxis.set_offset_position('left')
    cbar.update_ticks()

    axes[1].set_title("true image")
    if true is not None:
        im1 = axes[1].imshow(true)
        cbar = plt.colorbar(im1, ax=axes[1])
        # Set the colorbar tick format
        cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((-4, 4))
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()
    if clean_sources is not None:
        axes[1].scatter(clean_sources[:, 0], clean_sources[:, 1], s=90, c='none', marker="o", edgecolor='r',
                        linewidths=1)
    axes[1].set_xticks([])
    axes[1].set_yticks([])


    if true is None or np.sum(true - generated) != 0:
        axes[2].set_title("predicted image")
        im2 = axes[2].imshow(generated)
        if generated_sources is not None:
            axes[2].scatter(generated_sources[:, 0], generated_sources[:, 1], s=90, c='none', marker="o", edgecolor='r',
                            linewidths=1)
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        cbar = plt.colorbar(im2, ax=axes[2])
        # Set the colorbar tick format
        cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((-4, 4))
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()

    if uncertainty is not None and np.sum(uncertainty) == 0:
        pass
    else:
        if uncertainty is None:
            axes[3].set_title("Difference")
            im3 = axes[3].imshow(true - generated, cmap="plasma")
        else:
            axes[3].set_title("uncertainty")
            im3 = axes[3].imshow(uncertainty, cmap="plasma")

        cbar = plt.colorbar(im3, ax=axes[3])
        # Set the colorbar tick format
        cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((-6, 6))
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()
        axes[3].set_xticks([])
        axes[3].set_yticks([])

    plt.suptitle(f"Results")
    if save_fig:
        plt.savefig(save_name)
    else:
        plt.show()
def round_custom(x):
    return np.round(x * 2) / 2

ROUND_SNR = 0


class PredictedCatalog:
    def __init__(
            self,
            folder,
            dataset_folder,
            runs_per_sample,
            image_size=512,
            eps=5e-5,
            partition="test"
    ):
        self.folder = folder
        self.dataset_folder = dataset_folder
        self.runs_per_sample = runs_per_sample
        self.image_size = image_size
        self.eps = eps
        self.partition = partition
        self.power = self._extract_power_from_folder_name()

        self.generated_images = []
        self.images = []
        self.sky_indexes = []
        self.noisy_input = []
        self.sky_keys = []
        self.phase_dict = {}
        self.snr_extended = {}
        self.load_data()

    def _extract_power_from_folder_name(self):
        # Search for a pattern 'power' followed by one or more digits possibly with a decimal point
        match = re.search(r'power(\d+(\.\d+)?)', self.folder)
        # If a match is found, extract the power value, otherwise set to default (30)
        return float(match.group(1)) if match else 30

    def _compute_batch_nbs(self):
        file_prefix = "batch="
        file_suffix = "_test_dirty_noisy.npy"
        all_files = [f for f in os.listdir(self.folder) if f.startswith(file_prefix) and f.endswith(file_suffix)]
        # Extract batch numbers
        batch_numbers = [int(f[len(file_prefix):-len(file_suffix)]) for f in all_files]
        batch_numbers.sort()
        return batch_numbers

    def reorder_repeated(self, input_array,):
        im_shape = input_array.shape[1:]
        N = input_array.shape[0]

        # Compute the number of times each almost same image appears in the input array
        num_repeats = N // self.runs_per_sample

        # Reshape the input array to stack the almost same images one after another
        reshaped_array = np.reshape(input_array, (self.runs_per_sample, num_repeats, *im_shape))

        # Transpose the dimensions of the reshaped array to put the almost same images together
        transpose_axes = [1, 0] + list(range(2, len(im_shape) + 2))
        reshaped_array = np.transpose(reshaped_array, transpose_axes)

        # Reshape the array back to its original shape
        reordered_array = np.reshape(reshaped_array, (N, *im_shape))

        return reordered_array
    def load_data(self):
        if self.runs_per_sample != -1:
            batch_numbers = self._compute_batch_nbs()
            for i in batch_numbers:
                line=f"batch={i}_"
                test_generated_images_i = np.load(f"{self.folder}/{line}{self.partition}_generated_images.npy")
                test_images_i = np.load(f"{self.folder}/{line}{self.partition}_images.npy")
                sky_indexes_i = np.load(f"{self.folder}/{line}{self.partition}_sky_indexes.npy")
                noisy_input_i = np.load(f"{self.folder}/{line}{self.partition}_dirty_noisy.npy")

                test_generated_images_i = self.reorder_repeated(test_generated_images_i,)
                test_images_i = self.reorder_repeated(test_images_i,)
                sky_indexes_i = self.reorder_repeated(sky_indexes_i,)
                noisy_input_i = self.reorder_repeated(noisy_input_i,)

                self.generated_images.append(test_generated_images_i)
                self.images.append(test_images_i)
                self.sky_indexes.append(sky_indexes_i)
                self.noisy_input.append(noisy_input_i)

            self.generated_images = np.concatenate(self.generated_images)
            self.images = np.concatenate(self.images)
            self.sky_indexes = np.concatenate(self.sky_indexes)
            self.noisy_input = np.concatenate(self.noisy_input)

        else:
            self.generated_images = np.load(f"{folder}/{partition}_generated_images.npy")
            self.images = np.load(f"{folder}/{partition}_images.npy")
            self.sky_indexes = np.load(f"{folder}/{partition}_gt_sources.npy")
            self.noisy_input = np.load(f"{folder}/{partition}_dirty_noisy.npy")

        self.sky_keys = np.load(f"{self.dataset_folder}/sky_keys.npy")
        self.phase_dict = np.load(f"{self.dataset_folder}/ra_dec.npy", allow_pickle=True).item()

        self.noisy_folder = f"{self.dataset_folder}/dirty"
        self.true_folder = f"{self.dataset_folder}/true"

        self.noisy_im_filenames = os.listdir(self.noisy_folder)
        self.noisy_im_filenames.sort()

        # ra, dec, SNR, SNR normalized, flux, major, minor
        self.additional_line=""

    def load_header(self, key):
        # our basic train val and test sets. All headers are saved separately as pickle objects
        # in the separate folder "headers"
        # Create a new header instance
        values = self.phase_dict.get(key)
        header = fits.Header()

        header['NAXIS'] = 4
        header['CTYPE1'] = 'RA---SIN'
        header['CRPIX1'] = self.image_size / 2
        header['CRVAL1'] = values["RA"]
        header['CUNIT1'] = 'deg'
        header['CDELT1'] = -2.777777777778E-05
        header['CTYPE2'] = 'DEC--SIN'
        header['CRPIX2'] = self.image_size / 2
        header['CRVAL2'] = values["DEC"]
        header['CUNIT2'] = 'deg'
        header['CDELT2'] = 2.777777777778E-05
        return header

    def load_wcs(self, key):
        header = self.load_header(key)
        wcs = WCS(header)
        return wcs

    def run_test_experiment(
            self,
            i,
            verbose=False,
            visualization=False,
            apply_itransform=True,
            save_fig=False,
            aggr="median",
    ):
        # corresponding index from fits name
        sky_index = self.sky_indexes[i]
        key = self.sky_keys[sky_index]
        noisy_im = im_reshape(self.noisy_input[i])  # np.load(noisy_folder + "/" + noisy_im_filenames[sky_index])
        repeat_images = self.runs_per_sample
        save_folder = self.folder

        # default values if no sources are found
        data = {}
        data["predicted_sources"] = []
        data["predicted_sources_from_true"] = []
        data["gt_sources"] = []


        if verbose or visualization:
            print("============================================FINAL============================================")
            print(i)

        # get pixel true coordinates
        wcs = self.load_wcs(key)

        gen_ims = []
        sources_from_generated_all_runs = []
        reconstruction_metrics_generated_all_runs = []

        for j in range(repeat_images):
            gen_im = self.generated_images[i + j]
            gen_im = gen_im[:, :, 0]
            gen_im = im_reshape(gen_im)
            if np.max(true_itrasnform(gen_im, self.power)) < 1e-10:
                continue

            # per image detection cycle

            if apply_itransform:
                gen_im_astro = true_itrasnform(gen_im, self.power)

            sources_from_generated = image_to_sources(gen_im_astro)
            sources_from_generated = pix_to_astro(sources_from_generated, wcs)
            # adding column for j th run of diffusion model
            sources_from_generated = add_column_i(sources_from_generated, j)

            if sources_from_generated is not None:
                sources_from_generated_all_runs.append(sources_from_generated)

            gen_ims.append(gen_im)
        if len(gen_ims) == 0:
            gen_ims = [gen_im, ]

        ims_array = np.array(gen_ims)  # convert list to numpy array
        gen_im = aggregate_images(ims_array, aggregation=aggr)
        uncertainty = np.std(ims_array, axis=0)  # /np.mean(ims_array, axis=0)

        if apply_itransform:
            gen_im_astro = true_itrasnform(gen_im, self.power)

        sources_from_generated_pix = image_to_sources(gen_im_astro)
        sources_from_generated = pix_to_astro(sources_from_generated_pix, wcs)

        # -1 to flag that it is aggregated
        sources_from_generated = add_column_i(sources_from_generated, -1)

        if sources_from_generated is not None:
            sources_from_generated_all_runs.append(sources_from_generated)

        if verbose:
            # Create a DataFrame
            if sources_from_generated is not None:
                df = pd.DataFrame(sources_from_generated, columns=astropy_columns + ["diffusion_idx", ])
                display(df)
            else:
                print("No sources from generated")

        if verbose or visualization:
            plot_generated_images(
                None,
                true_trasnform(gen_im_astro, self.power),
                noisy_im,
                uncertainty,
                sources=[None, sources_from_generated_pix, None],
                # sources=[gt_sources*gen_im.shape[1]/512, sources_from_true, gt_sources],
                save_fig=save_fig,
                save_name=f"{save_folder}/sample_{self.partition}_{i}{self.additional_line}.png",
            )
            print("=============================================================================================")
        if len(sources_from_generated_all_runs) > 0:
            sources_from_generated_all_runs = np.vstack(sources_from_generated_all_runs)
        else:
            sources_from_generated_all_runs = None

        data["predicted_sources"] = add_column_i(sources_from_generated_all_runs, i//self.runs_per_sample)
        return data

    def test(
            self,
            nb_sources=None,
            verbose=False,
            visualization=False,
            apply_itransform=True,
            SNR_fixed=None,
            plot_brightness=False,
            SNR_normalized=False,
            aggr="median",
    ):

        predicted_sources = []
        gt_sources = []
        predicted_sources_from_true = []
        reconstruction_metrics = []

        data = {}

        for i in tqdm.tqdm(range(0, len(self.images), self.runs_per_sample)):  #
            data = self.run_test_experiment(
                i,
                True,#verbose,
                True,#visualization,
                apply_itransform,
                plot_brightness,
                aggr=aggr
            )

            if not isinstance(data, dict):
                print(i)
                continue
            else:
                pass

            if data["predicted_sources"] is not None and len(data["predicted_sources"]) > 0:
                predicted_sources.append(data["predicted_sources"])

        if len(predicted_sources) > 0:
            predicted_sources = np.vstack(predicted_sources, )
        data["predicted_sources"] = predicted_sources

        return data

def aggregate_images(images, aggregation='mean'):
    if aggregation == 'mean':
        return np.mean(images, axis=0)
    if aggregation == 'median':
        return np.median(images, axis=0)
    if aggregation == 'medoid':
        if len(images) == 0:
            return None
        n = len(images)
        # Calculate pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(images[i].flatten() - images[j].flatten())
        # Sum the distances
        sum_distances = np.sum(distances, axis=1)
        # Find the index of the medoid image
        medoid_idx = np.argmin(sum_distances)
        # Return the medoid image
        return images[medoid_idx]
    else:
        print("Such aggregation method is not implemented:", aggregation)
        return None

def update_dict(
    filtered_predicted,
    final_metrics,
    gt_sources,
    key
    ):
    result, unmatched_nb = merge_pd_frames(gt_sources, filtered_predicted)
    index_matched = result.index_predicted[:-unmatched_nb]
    ##############################
    # Metrics for entire dataset
    ##############################
    metrics_keys = ["purity", "completeness", "f1", "tp", "fp", "fn", "f1", "values"]

    tp, fp, fn = compute_localization(result)
    purity, completeness, f1= compute_purity_completeness_f1(tp, fn, fp)
    values = result.dropna()

    for metrics_key in metrics_keys:
        final_metrics[key][metrics_key] = locals()[metrics_key]
    ##############################
    # Per SNR metrics
    ##############################
    metrics_keys = ["purity", "completeness", "nb_points", "snrs", "tp", "fp", "fn", "f1", "values"]
    metrics_sources_aggregated_per_snr = {metrics_key: [] for metrics_key in metrics_keys}

    # Iterating over SNR values to compute metrics
    for snrs in np.arange(1.0, 10.1, 0.5):
        snrs_low=snrs-0.25
        snrs_up=snrs+0.25

        # Filtering ground truth sources based on SNR bounds
        filtered_gt_sources = gt_sources[gt_sources["SNR"].between(snrs_low, snrs_up)]
        result, _ = merge_pd_frames(filtered_gt_sources, filtered_predicted)

        # Computing localization metrics
        tp, fp, fn = compute_localization(
            result,
            filtering_for_fp=result["SNR_estimated_predicted"].between(
                snrs_low,
                snrs_up
            ) & ~result["index_predicted"].isin(index_matched[:-unmatched_nb])
        )
        purity, completeness, f1= compute_purity_completeness_f1(tp, fn, fp)

        nb_points = len(filtered_gt_sources)
        values = result.dropna()

        # Aggregating metrics
        for metrics_key in metrics_keys:
            metrics_sources_aggregated_per_snr[metrics_key].append(locals()[metrics_key])

    final_metrics[key]["snr"] = metrics_sources_aggregated_per_snr.copy()
    ##############################
    # Per normalized SNR metrics
    ##############################
    metrics_keys = ["purity", "completeness", "nb_points", "snrs", "tp", "fp", "fn", "f1", "values"]
    metrics_sources_aggregated_per_snr = {metrics_key: [] for metrics_key in metrics_keys}

    for snrs in np.arange(1.0, 7.1, 0.5):
        snr_norm_low=snrs-0.25
        snr_norm_up=snrs+0.25
        filtered_gt_sources = gt_sources[gt_sources["SNR normalized"].between(snr_norm_low, snr_norm_up)]
        #filtered_predicted = filtered_predicted[filtered_predicted["SNR_norm_estimated"].between(snr_norm_low, snr_norm_up)]
        result, _ = merge_pd_frames(filtered_gt_sources, filtered_predicted)
        tp, fp, fn = compute_localization(
            result,
            filtering_for_fp=result["SNR_norm_estimated_predicted"].between(
                snr_norm_low,
                snr_norm_up
            ) & ~result["index_predicted"].isin(index_matched[:-unmatched_nb])
        )
        purity, completeness, f1= compute_purity_completeness_f1(tp, fn, fp)

        nb_points = len(filtered_gt_sources)
        values = result.dropna()

        # Aggregating metrics
        for metrics_key in metrics_keys:
            metrics_sources_aggregated_per_snr[metrics_key].append(locals()[metrics_key])

    final_metrics[key]["snrnorm"] = metrics_sources_aggregated_per_snr

    # Count the number of sources per image_idx

    # Defining metrics keys
    metrics_keys = ["purity", "completeness", "tp", "fp", "fn", "f1", "values"]
    metrics_nb_sources = {metrics_key: [] for metrics_key in metrics_keys}

    for fixed_sources_nb in [1,2,3,4,5]:
        # Counting sources per image index
        source_counts = gt_sources.groupby('image_idx').size()

        # Selecting image indices that have a fixed number of sources
        images_with_fixed_sources = source_counts[source_counts == fixed_sources_nb].index

        # Filtering data based on these image indices
        filtered_data = filtered_predicted[filtered_predicted['image_idx'].isin(images_with_fixed_sources)]
        filtered_gt_sources = gt_sources[gt_sources['image_idx'].isin(images_with_fixed_sources)]

        # Merging data frames for computing metrics
        result, _ = merge_pd_frames(filtered_gt_sources, filtered_data)

        # Computing localization metrics
        tp, fp, fn = compute_localization(result)

        # Computing purity, completeness, and F1 score
        purity, completeness, f1 = compute_purity_completeness_f1(tp, fn, fp)

        values = result.dropna()

        # Aggregating metrics
        for metrics_key in metrics_keys:
            metrics_nb_sources[metrics_key].append(locals()[metrics_key])

    final_metrics[key]["of_nbs"] = metrics_nb_sources
    return final_metrics


def compute_purity_completeness_f1(tp, fn, fp):
    purity = compute_metrics(tp, fp)
    completeness = compute_metrics(tp, fn)
    if purity == completeness == 0:
        f1 = 0
    else:
        f1 = 2 * completeness * purity / (purity + completeness)
    return purity, completeness, f1


def compute_metrics(tp, fp, ):
    if tp + fp != 0:
        metrics = tp / (tp + fp)
    else:
        if tp == 0:
            metrics = 1
        else:
            metrics = 0
    return metrics

def main(folders, dataset_folder, runs_per_sample, image_size=512, eps=5e-5, partition="test", consistent_ra_dec=False):
    for folder in folders:
        catalog = PredictedCatalog(folder, dataset_folder, runs_per_sample, image_size, eps, partition)
        reconstruction_columns = ["l2", "l1", "psnr", "ssim"]

        for aggr in ["mean", "medoid", "median"]:
            current_key = aggr

            final_metrics = initialize_final_metrics(runs_per_sample)
            res = catalog.test(plot_brightness=True,verbose=False, aggr=current_key)

            #predicted tables for sources
            predicted_sources = pd.DataFrame(res["predicted_sources"], columns=astropy_columns+["diff_idx",]+["image_idx",])
            predicted_sources["SNR_estimated"] = predicted_sources["source_sum"]/NOISE_PER_IMAGE
            predicted_sources["s_min"] = predicted_sources["semiminor_sigma"]
            predicted_sources["s_max"] = predicted_sources["semimajor_sigma"]
            #snr normalized
            beam_major = 0.89
            beam_minor = 0.82
            s_min = predicted_sources["s_min"]
            s_max = predicted_sources["s_max"]
            predicted_sources["SNR_norm_estimated"] = predicted_sources["SNR_estimated"]*(beam_major*beam_minor)/np.sqrt((beam_major**2+s_max**2)*(beam_minor**2+s_min**2))
            predicted_sources.to_csv(folder+f"/{current_key}_predicted_sources.csv")
            predicted_sources[predicted_sources["diff_idx"]==-1].to_csv(folder + f"/{current_key}_predicted_sources.csv")


        # Apply the function for each image_idx and concat the result
        if not consistent_ra_dec:
            try:
                clustered_data = pd.concat([cluster_points(predicted_sources, image_idx, eps) for image_idx in predicted_sources["image_idx"].unique()])
            except ValueError:
                return
        else:
            try:
                clustered_data = cluster_points(predicted_sources, None, eps)
            except ValueError:
                return


        # Group by image_idx and cluster, and calculate mean and std for each group
        aggregated_data = clustered_data.groupby(["image_idx", "cluster"]).agg({
            "source_sum": ["mean", "std"],
            "ra": ["mean", "std"],
            "dec": ["mean", "std"],
            "s_max": ["mean", "std"],
            "s_min": ["mean", "std"],
            "orientation": ["mean", "std"],
            "eccentricity": ["mean", "std"],
            "max_value": ["mean", "std"],
            "equivalent_radius": ["mean", "std"],
            "area": ["mean", "std"],
            "SNR_norm_estimated": ["mean", "std"],
            "SNR_estimated": ["mean", "std"],
            "diff_idx": "count"  # Count how many times source was present out of 20
        })

        # Rename the columns for clarity
        aggregated_data.columns = ["_".join(col) for col in aggregated_data.columns]

        aggregated_sources = aggregated_data.copy()
        # Rename columns with *_mean to the original name
        final_columns = {}
        for column in aggregated_sources.columns:
            if column.endswith("_mean"):
                final_columns[column] = column[:-5]  # Remove '_mean' from the column name
            else:
                final_columns[column] = column

        aggregated_sources.rename(columns=final_columns, inplace=True)

        # Reset the index
        aggregated_sources.reset_index(inplace=True)
        aggregated_sources.to_csv(folder+"/aggregated_sources.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        default='./configs/generator.yaml',
                        help='Path to config')
    parser.add_argument('--runs_per_sample',
                        type=int, default=-1,
                        help='Runs per images passed for generating.')
    parser.add_argument('--consistent_ra_dec',
                        type=bool, default=False,
                        help='If set to true, data assumes to be self consistent, \
                        so each source is searched among all images.\
                        The changes will be seen only for aggregated sources via detect-aggregate.')
    parser.add_argument('--folders', nargs='+', required=True, help='List of folders to process')
    args = parser.parse_args()
    config = get_config(args.config)

    folders = args.folders
    main(
        folders,
        dataset_folder=config["dataset"]["image_path"],
        runs_per_sample=args.runs_per_sample,
        consistent_ra_dec=args.consistent_ra_dec
    )