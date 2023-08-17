#!/usr/bin/env python
# coding: utf-8

# In[585]:


import torch
import torchvision
import os


import numpy as np
import matplotlib.pyplot as plt

from glob import glob

from scipy import signal

import seaborn as sns
from photutils import detect_sources
from astropy.convolution import Gaussian2DKernel, convolve

from astropy.stats import sigma_clipped_stats

import matplotlib.ticker as ticker

import pandas as pd
from IPython.display import display

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy
import warnings

from astropy.wcs import WCS
from astropy.wcs.wcs import FITSFixedWarning
import warnings

import pickle
# ignore FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

def prepareHeaders(header):
    header.remove("HISTORY", remove_all=True)
    header.remove("OBJECT", remove_all=True)
    header.remove("DATE", remove_all=True)
    header.remove("DATE-OBS", remove_all=True)
    header.remove("TIMESYS", remove_all=True)
    header.remove("ORIGIN", remove_all=True)
    return header


# In[785]:


dataset_folder = "/home/drozdova/projects/dataset_ska/"
folder = "/home/drozdova/projects/ska-project/galaxy_zoo_generation_diffusion/results/direct_mag_ph_simple_512/"
folder = "/home/drozdova/projects/ska-project/diffusion-true-ska/results/repeat_128_917/"
partition="val"

folders = [
    #"/home/drozdova/projects/ska-project/galaxy_zoo_generation_diffusion/results/direct_mag_ph_simple/",
    #"/home/drozdova/projects/ska-project/galaxy_zoo_generation_diffusion/results/direct_re_im_simple_294/",
    
    #"/home/drozdova/projects/ska-project/diffusion-true-ska/results/direct_nocond_uv/",   
    "/home/drozdova/projects/ska-project/diffusion-true-ska/results/repeat_nocond_uv/",  
    
    "/home/drozdova/projects/ska-project/diffusion-true-ska/results/direct_nocond_uv_stable/",
    "/home/drozdova/projects/ska-project/diffusion-true-ska/results/repeat_nocond_uv_stable/",  

    "/home/drozdova/projects/ska-project/diffusion-true-ska/results/direct_nocond",
    "/home/drozdova/projects/ska-project/diffusion-true-ska/results/repeat_128_917/",
    
    "/home/drozdova/projects/ska-project/galaxy_zoo_generation_diffusion/results/direct_mag_ph_simple_512/",
    "/home/drozdova/projects/ska-project/galaxy_zoo_generation_diffusion/results/direct_re_im_simple_512/",
    
    #"/home/drozdova/projects/ska-project/galaxy_zoo_generation_diffusion/results/direct",
    #"/home/drozdova/projects/ska-project/diffusion-true-ska/results/repeat_nocond",

    

    #"/home/drozdova/projects/ska-project/diffusion-true-ska/results/direct_128_640",
    #"/home/drozdova/projects/ska-project/diffusion-true-ska/results/direct_128",
    #"/home/drozdova/projects/ska-project/diffusion-true-ska/results/repeat_128_917",
    
    #"/home/drozdova/projects/ska-project/galaxy_zoo_generation_diffusion/results/repeat",
]
for folder in folders:
    def reorder_repeated(input_array, repeat=20):
        im_shape = input_array.shape[1:]
        N = input_array.shape[0]

        # Compute the number of times each almost same image appears in the input array
        num_repeats = N // repeat

        # Reshape the input array to stack the almost same images one after another
        reshaped_array = np.reshape(input_array, (repeat, num_repeats, *im_shape))

        # Transpose the dimensions of the reshaped array to put the almost same images together
        transpose_axes = [1, 0] + list(range(2, len(im_shape) + 2))
        reshaped_array = np.transpose(reshaped_array, transpose_axes)

        # Reshape the array back to its original shape
        reordered_array = np.reshape(reshaped_array, (N, *im_shape))

        return reordered_array

    if folder.find("repeat") != -1:


        test_generated_images = []
        test_images = []
        test_labels1 = []
        test_labels2 = []
        sky_indexes = []
        noisy_input = []

        repeat_images = 20

        for i in range(0,30):
            print(i)
            try:
            	test_generated_images_i = np.load(f"{folder}/batch={i}_{partition}_generated_images.npy")
            except FileNotFoundError:
                continue
            test_images_i = np.load(f"{folder}/batch={i}_{partition}_images.npy")
            test_labels1_i = np.load(f"{folder}/batch={i}_{partition}_labels1.npy")
            test_labels2_i = np.load(f"{folder}/batch={i}_{partition}_labels2.npy")
            sky_indexes_i = np.load(f"{folder}/batch={i}_{partition}_gt_sources.npy")
            noisy_input_i = np.load(f"{folder}/batch={i}_{partition}_dirty_noisy.npy")

            test_generated_images_i = reorder_repeated(test_generated_images_i)
            test_images_i = reorder_repeated(test_images_i)
            test_labels1_i = reorder_repeated(test_labels1_i)
            test_labels2_i = reorder_repeated(test_labels2_i)
            sky_indexes_i = reorder_repeated(sky_indexes_i)
            noisy_input_i = reorder_repeated(noisy_input_i)

            test_generated_images.append(test_generated_images_i)
            test_images.append(test_images_i)
            test_labels1.append(test_labels1_i)
            test_labels2.append(test_labels2_i)
            sky_indexes.append(sky_indexes_i)
            noisy_input.append(noisy_input_i)

        test_generated_images = np.concatenate(test_generated_images)
        test_images = np.concatenate(test_images)

        test_labels1 = np.concatenate(test_labels1)
        test_labels2 = np.concatenate(test_labels2)
        sky_indexes = np.concatenate(sky_indexes)
        noisy_input = np.concatenate(noisy_input)

    else:
        repeat_images=1
        test_generated_images = np.load(f"{folder}/{partition}_generated_images.npy")
        test_images = np.load(f"{folder}/{partition}_images.npy")

        test_labels1 = np.load(f"{folder}/{partition}_labels1.npy")
        test_labels2 = np.load(f"{folder}/{partition}_labels2.npy")

        sky_indexes = np.load(f"{folder}/{partition}_gt_sources.npy")

        noisy_input = np.load(f"{folder}/{partition}_dirty_noisy.npy")

    mean_vectors = np.load(f"{dataset_folder}/data/mean_vectors.npy")
    std_vectors = np.load(f"{dataset_folder}/data/std_vectors.npy")

    sky_sources = np.load(f"{dataset_folder}/data/sky_sources.npy", allow_pickle=True)
    sky_keys = np.load(f"{dataset_folder}/sky_keys.npy")

    noisy_folder = f"{dataset_folder}/data/dirty_noisy_wo_processing"
    noisy_im_filenames = os.listdir(noisy_folder)
    noisy_im_filenames.sort()

    snr_extended = np.load(f"{dataset_folder}/sky_sources_snr_extended.npy", allow_pickle=True)
    snr_extended = snr_extended.item()
    #ra, dec, SNR, SNR normalized, flux, major?, minor?

    true_folder = f"{dataset_folder}/data/true_wo_processing"

    additional_line=""
    #additional_line="_test=gen"
    #test_generated_images = test_images


    # In[ ]:





    # In[ ]:





    # In[786]:


    #l = []
    #for key in snr_extended:
    #    line = snr_extended[key]
    #    l.append(line[0][2])
    #plt.hist(l)


    # In[787]:


    def image_to_sources(xp, verbose=False):
        # Smooth the image with a Gaussian kernel to help with source detection
        kernel = Gaussian2DKernel(1, x_size=1, y_size=1)
        smoothed = convolve(xp, kernel)

        smoothed = xp
        # Detect sources using the smoothed image
        threshold = 0.1 * np.std(smoothed)
        segm = detect_sources(smoothed, threshold, npixels=1)

        # Extract the source properties
        mean, median, std = sigma_clipped_stats(xp, sigma=1)
        props = source_properties(xp - median, segm, background=median)

        coords = []
        # Print the properties of each detected source
        for prop in props:
            if verbose:
                print('Xcentroid =', prop.xcentroid.value)
                print('Ycentroid =', prop.ycentroid.value)
                print('Source area =', prop.area.value)
                print('Source integrated flux =', prop.source_sum)
            coords.append([
                prop.xcentroid.value,
                prop.ycentroid.value,
                prop.source_sum,
                prop.area.value
            ])
        coords = np.array(coords)
        coords_sorted = coords[np.argsort(coords[0, 1])]
        return coords


    # In[788]:



    def true_trasnform(true):
        true = (true)**(1./30)
        true = (true)/(0.7063881)
        true = (true - 0.5)/0.5
        return true

    def true_itrasnform(true):
        true_back= true*(0.7063881)
        true_back = (true_back)**(30)
        return true_back

    def noisy_trasnform(noisy):
        noisy = noisy/5.0e-4
        noisy = (noisy - 0.5) / 0.5
        return noisy

    def noisy_itrasnform(noisy):
        noisy = noisy*0.5+0.5
        noisy= noisy*(5.0e-4)
        return noisy


    # In[789]:


    def match_points(coords_from_true, coords_from_gen, eps, idx=2, fixed_snr=None, snr_idx=3):

        if fixed_snr is not None:
            # need to apply filtering
            coords_from_true = coords_from_true[coords_from_true[:, snr_idx].round() == fixed_snr]
            if coords_from_true.shape[0] == 0:
                return None, 0, 0, 0
        if coords_from_gen is None:
            return None, 0, 0, len(coords_from_true)
        # Calculate distance between all pairs of points
        dist_matrix = np.sqrt(np.sum((coords_from_true[:, :2, np.newaxis] - coords_from_gen[:, :2, np.newaxis].T)**2, axis=1))
        # Find nearest neighbor for each point from coords_from_gen
        nearest_neighbors = np.argmin(dist_matrix, axis=0)
        nearest_neighbors_dist = np.min(dist_matrix, axis=0)

        # to make sure each true source has only one assignment
        sources_free = np.ones((coords_from_true.shape[0],))

        # Combine points from coords_from_gen that are close to each other and also close to same point(s) from coords_from_true
        combined_coords = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for i in range(len(coords_from_gen)):
            if nearest_neighbors_dist[i] < eps and sources_free[nearest_neighbors[i]] == 1:
                sources_free[nearest_neighbors[i]] = 0
                combined_coords.append([coords_from_true[nearest_neighbors[i], idx], coords_from_gen[i, idx]])
                true_positives += 1
            else:
                false_positives += 1

        # Count false negatives as the number of unmatched points in coords_from_true
        false_negatives = len(coords_from_true) - true_positives
        if (false_negatives) < 0:
            print(coords_from_true)
            print(coords_from_gen)
            print(true_positives)
        assert false_negatives >= 0 

        return np.array(combined_coords), true_positives, false_positives, false_negatives


    # In[790]:


    from skimage.transform import resize
    import astropy.units as u



    from photutils.segmentation import deblend_sources
    def image_to_sources(image, verbose=False, image_size=512):
        # Smooth the image with a Gaussian kernel to help with source detection
        if image_size == 128:
            kernel_size = 1
            kernel_x_size = 1
            kernel_y_size = 1
            sigma_clipped_stats_sigma = 3.0
            detect_npixels = 5
            deblend_sources_npixels = 1
            std_const = 5

        if image_size == 512:
            kernel_size = 1
            kernel_x_size = 3
            kernel_y_size = 3        
            sigma_clipped_stats_sigma = 20.0#10.0
            detect_npixels = 3
            deblend_sources_npixels = 5
            std_const = 50#12

        image = image
        kernel = Gaussian2DKernel(kernel_size,)# x_size=kernel_x_size, y_size=kernel_y_size)

        smoothed = convolve(image, kernel, boundary='wrap')

        # Calculate the threshold for source detection
        mean, median, std = sigma_clipped_stats(smoothed, sigma=sigma_clipped_stats_sigma)
        threshold =  (std_const* std)

        # Detect the sources in the image
        segm = detect_sources(smoothed, threshold, npixels=detect_npixels)

        # Deblend the sources
        segm_deblend = deblend_sources(smoothed, segm, npixels=deblend_sources_npixels)#, deblend_cont=0.01)

        # Extract the properties of the sources
        props = source_properties(smoothed, segm_deblend, error=None, background=None)


        coords = []
        # Print the properties of each detected source
        for prop in props:
            if verbose:
                print('Xcentroid =', prop.xcentroid.value)
                print('Ycentroid =', prop.ycentroid.value)
                print('Source area =', prop.area.value)
                print('Source integrated flux =', prop.source_sum)
            coords.append([
                prop.xcentroid.value,
                prop.ycentroid.value,
                prop.source_sum,
                prop.area.value,
                prop.equivalent_radius.value,
                prop.semimajor_axis_sigma.value,
                prop.semiminor_axis_sigma.value,
                prop.max_value,
                prop.min_value,
                (prop.eccentricity*u.m).value,
                (prop.orientation*u.m).value,
            ])
        coords = np.array(coords)
        coords_sorted = coords[np.argsort(coords[0, 1])]


        return coords

    def flatten(l):
        return [[item[0],item[1]] for sublist in l for item in sublist]


    # In[792]:


    from photutils.segmentation import SourceCatalog


    def image_to_sources(
        image, verbose=False, 
        image_size=512, 
        detect_npixels=15
    ):
        if np.max(image) < 1e-10:
            return None #no sources
        # Smooth the image with a Gaussian kernel to help with source detection
        if image_size == 128:
            kernel_size = 1
            kernel_x_size = 1
            kernel_y_size = 1
            sigma_clipped_stats_sigma = 3.0
            detect_npixels = 5
            deblend_sources_npixels = 1
            std_const = 5

        if image_size == 512:
            kernel_size = 1
            kernel_x_size = 3
            kernel_y_size = 3        
            sigma_clipped_stats_sigma = 25.0#10.0
            detect_npixels = 10#detect_npixels
            deblend_sources_npixels = 10
            std_const = 120#12


        # Calculate the threshold for source detection
        mean, median, std = sigma_clipped_stats(image, sigma=sigma_clipped_stats_sigma)
        threshold =  (std_const* std)


        # Detect the sources in the image
        segm = detect_sources(image, threshold, npixels=detect_npixels)


        # Deblend the sources
        try:
            segm_deblend = deblend_sources(image, segm, npixels=deblend_sources_npixels, progress_bar=None)#, deblend_cont=0.01)
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
                0.1* 2.3548*prop.semiminor_sigma.value,
                0.1* 2.3548*prop.semimajor_sigma.value,

                prop.max_value,
                prop.min_value,
                (prop.eccentricity*u.m).value,
                (prop.orientation*u.m).value,
            ])
        coords = np.array(coords)
        coords_sorted = coords[np.argsort(coords[0, 1])]


        return coords

    def flatten(l):
        return [[item[0],item[1]] for sublist in l if sublist is not None for item in sublist]


    # In[793]:


    from functools import partial

    #for detect_npixels in range(1, 20, 2):
    #    print(detect_npixels)
    #    image_to_sources = partial(image_to_sources, detect_npixels=detect_npixels)
    #    res = test(plot_brightness=True,repeat_images=repeat_images)


    # In[794]:


    #detect_sources?


    # In[795]:


    astropy_columns = [
            "ra",#/xcentroid",
            "dec",#/ycentroid",
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

    def plot_generated_images_(
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
            if np.sum(true-generated) == 0:
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
                #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            else:
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
        elif np.sum(true-generated) != 0:
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18,4))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18,4))

        axes[0].set_title("noisy dirty image with true sources")
        im0 = axes[0].imshow(artificial_dirty_im)
        if true_sources is not None:
            axes[0].scatter(true_sources[:,0], true_sources[:,1], s=90, c='none', marker="o", edgecolor='r', linewidths=1)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        cbar = plt.colorbar(im0, ax=axes[0])
        # Set the colorbar tick format
        cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((-4, 4))
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()

        axes[1].set_title("true image with true sources")
        im1 = axes[1].imshow(true)
        if clean_sources is not None:
            axes[1].scatter(clean_sources[:,0], clean_sources[:,1], s=90,c='none', marker="o", edgecolor='r', linewidths=1)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        cbar = plt.colorbar(im1, ax=axes[1])
        # Set the colorbar tick format
        cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((-4, 4))
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()

        if np.sum(true-generated) != 0 or True:
            axes[2].set_title("true image with predicted sources")
            im2 = axes[2].imshow(generated)
            if generated_sources is not None:
                axes[2].scatter(generated_sources[:,0], generated_sources[:,1], s=90,c='none', marker="o", edgecolor='r', linewidths=1)
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
            plt.close()
        else:
            #plt.show()
            pass


    # In[799]:



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
            if np.sum(true-generated) == 0:
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
                #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            else:
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
        elif np.sum(true-generated) != 0:
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18,4))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18,4))

        axes[0].set_title("noisy dirty image")
        im0 = axes[0].imshow(artificial_dirty_im)
        if true_sources is not None:
            axes[0].scatter(true_sources[:,0], true_sources[:,1], s=90, c='none', marker="o", edgecolor='r', linewidths=1)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        cbar = plt.colorbar(im0, ax=axes[0])
        # Set the colorbar tick format
        cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((-4, 4))
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()

        axes[1].set_title("true image")
        im1 = axes[1].imshow(true)
        if clean_sources is not None:
            axes[1].scatter(clean_sources[:,0], clean_sources[:,1], s=90,c='none', marker="o", edgecolor='r', linewidths=1)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        cbar = plt.colorbar(im1, ax=axes[1])
        # Set the colorbar tick format
        cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((-4, 4))
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()

        if np.sum(true-generated) != 0 or True:
            axes[2].set_title("predicted image")
            im2 = axes[2].imshow(generated)
            if generated_sources is not None:
                axes[2].scatter(generated_sources[:,0], generated_sources[:,1], s=90,c='none', marker="o", edgecolor='r', linewidths=1)
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


    # In[800]:


    from scipy.ndimage import zoom
    from skimage.transform import resize


    def load_header(key="0a1a6090-3e23-4708-bde7-fd875f57f6c1"):
        pickle_path = f'{dataset_folder}/data/headers/dirty_gaussians_{key}.pickle'
        with open(pickle_path, 'rb') as f:
            header = pickle.load(f)
        return header

    def load_wcs(key):
        header = load_header(key)
        wcs = WCS(header)
        return wcs

    def actro_to_pix(true_sources, wcs):
        true_sources = np.array(true_sources)
        coords_deg = np.hstack((true_sources[:,0:1], true_sources[:,1:2], np.zeros((true_sources.shape[0], 2))))
        coords_pix = wcs.wcs_world2pix(coords_deg, 0)
        gt_sources = np.array(coords_pix)[:,:4]#*128/512
        gt_sources = gt_sources.astype(int).astype(float)
        return gt_sources

    def pix_to_astro(pixel_coords, wcs):
        pixel_coords = np.array(pixel_coords)
        coords_pix = np.hstack((pixel_coords[:,0:1], pixel_coords[:,1:2], np.zeros((pixel_coords.shape[0], 2))))
        coords_deg = wcs.wcs_pix2world(coords_pix, 0)
        celestial_coords = np.array(coords_deg)[:,:2]
        return celestial_coords

    def im_reshape(downsampled_array):
        if downsampled_array.shape[1] == 512:
            return downsampled_array
        im_size = downsampled_array.shape[1]
        scale_factor = 512//im_size
        downsampled_image = torch.tensor(downsampled_array).reshape(1,1,im_size,im_size,)
        upsampled_image = torch.nn.functional.interpolate(downsampled_image, scale_factor=scale_factor, mode='bicubic')
        upsampled_image = upsampled_image.data.numpy()[0,0,:,:,]
        return upsampled_image

    def run_test_experiment(
        i,
        nb_sources=None, 
        repeat_images=1,
        verbose=False, 
        visualization=False, 
        apply_itransform=True,
        SNR_fixed=None,
        plot_brightness=False,
        save_folder=folder,
        save_fig=False,
        full_return=False,
        SNR_normalized=False,
        zoom_nb=4,
    ):        
            # corresponding index from fits name
            sky_index = sky_indexes[i]
            noisy_im = np.load(noisy_folder + "/" + noisy_im_filenames[sky_index])
            im = np.load(true_folder + "/" + noisy_im_filenames[sky_index])
            im = np.nan_to_num(im)
            key = sky_keys[sky_index]
            true_sources = np.array(snr_extended[key])

            # default value if no sources are found
            data = {}
            data["tp"] = 0
            data["fp"] = 0
            data["fn"] = len(true_sources)
            data["brightness"] = None

            if nb_sources is not None:
                if len(true_sources) != nb_sources:
                    data["fn"] = 0
                    # in this case it should not contribute to fn as the whole image is excluded
                    return data


            if verbose or visualization:
                print("============================================FINAL============================================")
                print(i)



            #get pixel true coordinates
            wcs = load_wcs(key)
            gt_sources = actro_to_pix(true_sources, wcs)

            gt_sources = np.hstack(
                (
                    gt_sources[:,:2], 
                    true_sources[:,4].reshape(-1,1), #flux
                    true_sources[:,2].reshape(-1,1), #SNR
                    true_sources[:,3].reshape(-1,1), #SNR normalized
                    true_sources[:,5].reshape(-1,1), #major
                    true_sources[:,6].reshape(-1,1), #nimor
                )
            )

            # unpacking generated images
            gen_ims = []
            for j in range(repeat_images):
                gen_im = test_generated_images[i+j]
                gen_im = gen_im[:,:,0]
                gen_im = im_reshape(gen_im)
                gen_ims.append(gen_im)
            ims_array = np.array(gen_ims)  # convert list to numpy array
            gen_im = np.median(ims_array, axis=0)
            uncertainty=np.std(ims_array, axis=0)
            if apply_itransform:
                gen_im = true_itrasnform(gen_im)


            sources_from_true = image_to_sources(im)
            sources_from_generated = image_to_sources(gen_im)


            if visualization or save_fig:
                plot_generated_images(
                    im,
                    gen_im,
                    noisy_im, 
                    np.std(ims_array, axis=0)/np.mean(ims_array, axis=0), 
                    sources=[sources_from_true, sources_from_generated, gt_sources],
                    #sources=[gt_sources*gen_im.shape[1]/512, sources_from_true, gt_sources],
                    save_fig=save_fig,
                    save_name=f"{save_folder}/sample_{partition}_{i}{additional_line}.png",
                )

            # we do not have images with no sources in our datasets
            #TODO
            gt_sources[:,:2] = gt_sources[:,:2]*gen_im.shape[1]/512
            gt_sources_astro = gt_sources
            gt_sources_astro[:,:2] = pix_to_astro(gt_sources_astro, wcs)

            sources_from_generated_astro = sources_from_generated
            if sources_from_generated is not None:
                sources_from_generated_astro[:,:2] = pix_to_astro(sources_from_generated_astro, wcs)

            sources_from_true_astro = sources_from_true
            sources_from_true_astro[:,:2] = pix_to_astro(sources_from_true_astro, wcs)

            if verbose:    
                # Create a DataFrame
                df = pd.DataFrame(gt_sources, columns=casa_columns)
                display(df)

                # Create a DataFrame
                sources_from_generated[:,:2] =  sources_from_generated_astro[:,:2]
                df = pd.DataFrame(sources_from_generated, columns=astropy_columns)
                display(df)

                # Create a DataFrame
                df = pd.DataFrame(sources_from_true, columns=astropy_columns)
                display(df)

            if not SNR_normalized:
                br, tp, fp, fn = match_points(gt_sources_astro, sources_from_generated_astro, 5e-5, idx=2, fixed_snr=SNR_fixed)
                majors, _,_,_ = match_points(gt_sources_astro, sources_from_generated_astro, 5e-5, idx=5, fixed_snr=SNR_fixed)
                minors, _,_,_ = match_points(gt_sources_astro, sources_from_generated_astro, 5e-5, idx=6, fixed_snr=SNR_fixed)
            else:
                br, tp, fp, fn = match_points(gt_sources_astro, sources_from_generated_astro, 5e-5, idx=2, fixed_snr=SNR_fixed, snr_idx=4)
                majors, _,_,_ = match_points(gt_sources_astro, sources_from_generated_astro, 5e-5, idx=5, fixed_snr=SNR_fixed)
                minors, _,_,_ = match_points(gt_sources_astro, sources_from_generated_astro, 5e-5, idx=6, fixed_snr=SNR_fixed)


            if verbose:
                print(tp, fp, fn)
                print(br)

            if verbose or visualization:
                if len(sources_from_true)!=len(true_sources) or len(sources_from_generated)!=len(true_sources) :
                    print(sources_from_true)
                    print(sources_from_generated)        
                    print(true_sources) 

                    print(len(sources_from_true), len(sources_from_generated), len(true_sources))
                print("=============================================================================================")
            if full_return:
                data["tp"] = tp
                data["fp"] = fp
                data["fn"] = fn
                data["brightness"] = br
                data["majors"] = majors
                data["minors"] = minors
                data["gen_ims"] = gen_ims
                data["im"] = im
                data["uncertainty"] = uncertainty
                data["gen_im"] = gen_im
                data["noisy_im"] = noisy_im
                return data

            return tp, fp, fn, brightness
    def test(
        nb_sources=None, 
        repeat_images=1,
        verbose=False, 
        visualization=False, 
        apply_itransform=True,
        SNR_fixed=None,
        plot_brightness=False,
        save_fig=False,
        SNR_normalized=False,
    ):
        brightnesses = []
        majors = []
        minors = []

        TP = []
        FP = []
        FN = []

        data = {}

        for i in range(0,len(test_images), repeat_images):#
            data = run_test_experiment(
                i,
                nb_sources, 
                repeat_images,
                verbose, 
                visualization, 
                apply_itransform,
                SNR_fixed,
                plot_brightness,
                SNR_normalized=SNR_normalized,
                full_return=True,
            )
            if not isinstance(data, dict):
                continue
            tp, fp, fn, brightness = data["tp"], data["fp"], data["fn"], data["brightness"]

            TP.append(tp)
            FP.append(fp)
            FN.append(fn)
            if brightness is None:
                continue
            major = data["majors"]
            minor = data["minors"]
            brightnesses.append(brightness)
            majors.append(major)
            minors.append(minor)
        if len(brightnesses) > 0:
            brightnesses_array = np.array(flatten(brightnesses))
            majors = np.array(flatten(majors))
            minors = np.array(flatten(minors))
            #brightnesses_array = np.array(brightnesses)
            if plot_brightness:
                plt.figure(figsize=(15,10))

                error  =  5e-5
                x = np.linspace(np.min(brightnesses_array[:,0]), np.max(brightnesses_array[:,0]), 1000)
                plt.fill_between(
                    x, 
                    x - error, 
                    x + error, 
                    color='lightcoral',
                    interpolate=True,
                    alpha=0.5
                )
                plt.plot(x, x, color="red")
                plt.scatter(brightnesses_array[:,0], brightnesses_array[:,1], s=0.5)
                plt.xlabel("true flux")
                plt.ylabel("generated flux")
                plt.savefig(f"{folder}/brightness_{partition}_{additional_line}.jpg",)
                plt.close()

                error = 0

                plt.figure(figsize=(15,10))
                x = np.linspace(np.min(majors[:,0]), np.max(majors[:,0]), 1000)
                plt.fill_between(
                    x, 
                    x - error, 
                    x + error, 
                    color='lightcoral',
                    interpolate=True,
                    alpha=0.5
                )
                plt.plot(x, x, color="red")
                plt.scatter(majors[:,0], majors[:,1], s=1.5)
                plt.xlabel("true majors")
                plt.ylabel("generated majors")
                plt.savefig(f"{folder}/majors_{partition}_{additional_line}.jpg",)
                plt.close()

                plt.figure(figsize=(15,10))
                x = np.linspace(np.min(minors[:,0]), np.max(minors[:,0]), 1000)
                plt.fill_between(
                    x, 
                    x - error, 
                    x + error, 
                    color='lightcoral',
                    interpolate=True,
                    alpha=0.5
                )
                plt.plot(x, x, color="red")
                plt.scatter(minors[:,0], minors[:,1], s=1.5)
                plt.xlabel("true minors")
                plt.ylabel("generated minors")
                plt.savefig(f"{folder}/minors_{partition}_{additional_line}.jpg",)
                plt.close()
            data["brightness"] = brightnesses_array
            data["majors"] = majors
            data["minors"] = minors

        TP = np.array(TP)
        FP = np.array(FP)
        FN = np.array(FN)

        purity=np.sum(TP)/(np.sum(TP)+np.sum(FP))
        completeness=np.sum(TP)/(np.sum(TP)+np.sum(FN))

        print("Purity =",purity)
        print("Completeness =",completeness)
        print("Nb of samples =", np.sum(TP))

        if nb_sources is None and SNR_fixed is None:
            # Open the file in write mode
            with open(f"{folder}/results_{partition}{additional_line}.txt", "w") as f:
                # Write some data to the file
                f.write(f"Purity = {purity}\n")
                f.write(f"Completeness = {completeness}\n")
                f.write(f"Nb_of_samples = {np.sum(TP)}\n")

        data["purity"] = purity
        data["completeness"] = completeness
        data["nb_samples"] = np.sum(TP)

        return data



    # In[ ]:



    res = test(plot_brightness=True,repeat_images=repeat_images)
    np.save(f"{folder}/brightness_{partition}{additional_line}.npy", res["brightness"])
    np.save(f"{folder}/majors_{partition}{additional_line}.npy", res["majors"])
    np.save(f"{folder}/minors_{partition}{additional_line}.npy", res["minors"])


    # In[ ]:





    # In[ ]:


    purities, completenesses = [], []
    for i in range(1,6):
        print(f"Nb of sources = {i}")
        data = test(i, plot_brightness=False, visualization=False, repeat_images=repeat_images)
        p, c = data["purity"], data["completeness"]
        purities.append(p)
        completenesses.append(c)
    purities = np.array(purities)
    completenesses = np.array(completenesses)


    # In[ ]:


    np.save(f"{folder}/sample_{partition}_purities_nbs{additional_line}.npy", purities)
    np.save(f"{folder}/sample_{partition}_completenesses_nbs{additional_line}.npy", completenesses)


    # In[ ]:


    plt.plot(range(1, len(purities)+1), purities, marker='o')
    plt.xlabel("Number of sources")
    plt.ylabel("Purity")
    plt.title("Dependence of number of sources and purity")
    plt.xticks(range(1, len(purities)+1)) # set xticks to be integers only
    plt.savefig(f"{folder}/sample_{partition}_purity_nbs{additional_line}.png",)
    plt.close()
    plt.clf()
    plt.plot(range(1, len(completenesses)+1), completenesses, marker='o')
    plt.xlabel("Number of sources")
    plt.ylabel("Completeness")
    plt.title("Dependence of number of sources and completeness")
    plt.xticks(range(1, len(completenesses)+1)) # set xticks to be integers only
    plt.savefig(f"{folder}/sample_{partition}_completeness_nbs{additional_line}.png",)
    plt.close()


    # In[ ]:


    completenesses = []
    snrs = []
    for i in range(1,11):
        print(f"SNR={i}")
        data= test(SNR_fixed=i, plot_brightness=False, visualization=False,repeat_images=repeat_images)
        if "completeness" in data:
            c = data["completeness"]
            completenesses.append(c)
            snrs.append(i)
    completenesses = np.array(completenesses)


    # In[ ]:


    np.save(f"{folder}/sample_{partition}_snr{additional_line}.npy", completenesses)


    # In[ ]:


    plt.plot(snrs, completenesses, marker='o')
    plt.xlabel("SNR")
    plt.ylabel("Completeness")
    plt.title("Dependence of SNR and completeness")
    plt.xticks(range(1, len(completenesses)+1)) # set xticks to be integers only
    plt.savefig(f"{folder}/sample_{partition}_snr{additional_line}.png",)
    plt.close()


    # In[ ]:


    completenesses = []
    snrs = []
    for i in range(1,9):
        print(f"SNR={i}")
        data= test(SNR_fixed=i, SNR_normalized=True, plot_brightness=False, visualization=False,repeat_images=repeat_images)
        if "completeness" in data:
            c = data["completeness"]
            completenesses.append(c)
            snrs.append(i)
    completenesses = np.array(completenesses)
    np.save(f"{folder}/sample_{partition}_snrnorm{additional_line}.npy", completenesses)
    plt.plot(snrs, completenesses, marker='o')
    plt.xlabel("SNR")
    plt.ylabel("Completeness")
    plt.title("Dependence of SNR and completeness")
    plt.xticks(range(1, len(completenesses)+1)) # set xticks to be integers only
    plt.savefig(f"{folder}/sample_{partition}_snrnorm{additional_line}.png",)
    plt.close()


# In[ ]:



# In[ ]:


#import numpy as np
#import matplotlib.pyplot as plt
#


