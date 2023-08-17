'''
PyBDSF parameters for dirty images:

- noiseless:
    thresh_pix = 7
    thresh_isl = 5

- noisy:
    rms = 4.275E-05
'''
import os
import numpy as np
import bdsf
import math
import argparse
from astropy.io import fits
from astropy.table import Table
from pathlib import Path
from datetime import datetime

import warnings

warnings.simplefilter("ignore")

# ======================================================================================================================
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def applyPyBDSF(input_image, res_dir, args, ind=0, is_debug=False):
    if not args.rms_map:
        img = bdsf.process_image(input=input_image,
                                 clobber=True,  # Overwrite existing file?
                                 output_opts=False,
                                 quiet=True,
                                 advanced_opts=True,
                                 rms_map=False,
                                 rms_value=args.rms_value,
                                 )
    else:
        img = bdsf.process_image(input=input_image,
                                 clobber=True,
                                 output_opts=True,
                                 quiet=True,
                                 thresh=args.thresh,  # default = None
                                 thresh_pix=args.thresh_pix,  # default = 5,
                                 thresh_isl=args.thresh_isl,  # default = 3, ignored if  thresh="fdr"
                                 )

    # Write the source list catalog. File is named automatically.
    img.write_catalog(outfile=os.path.join(res_dir, "sources.cat"), format='ascii',
                      catalog_type='srl', clobber=True)

    if is_debug > 0:
        # Write the model image. Filename is specified explicitly.
        img.export_image(img_type='island_mask',
                         outfile=os.path.join(res_dir, "island_mask.fits"), clobber=True)

def prepareHeaders(header):
    header.remove("HISTORY", remove_all=True)
    header.remove("OBJECT", remove_all=True)
    header.remove("DATE", remove_all=True)
    header.remove("DATE-OBS", remove_all=True)
    header.remove("TIMESYS", remove_all=True)
    header.remove("ORIGIN", remove_all=True)

    return header

def pixelcoord(x, y, headers):
    pixel_x = headers['CRPIX1'] - np.round((headers['CRVAL1'] - x) / headers["CDELT1"]) - 1
    pixel_y = headers['CRPIX2'] - np.round((headers['CRVAL2'] - y) / headers["CDELT2"]) - 1

    return pixel_x, pixel_y

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def to_print(message):
    now = datetime.now()
    print(f"{now.strftime('%d.%m.%Y %H:%M:%S')}: {message}")

# ======================================================================================================================
parser = argparse.ArgumentParser(description='...')
parser.add_argument("--data_path", default="Python/PostDoc/datasets/AstroSignal/data_v.2/sampled_data/", type=str)

parser.add_argument("--sigma", default=1e-17)
parser.add_argument("--d", default=5e-5)
parser.add_argument("--n", default=100)

# pybdsf params
parser.add_argument("--thresh", default=None, choices=[None, "fdr", "hard"])
parser.add_argument("--thresh_pix", default=9, type=int)
# (a) should be smaller than thresh_pix
# (b) ignored if  thresh="fdr"
parser.add_argument("--thresh_isl", default=7, type=int)
# or
parser.add_argument("--rms_map", default=False, type=str2bool)
parser.add_argument("--rms_value", default=0.03, type=float)

# ======================================================================================================================
if __name__ == "__main__":
    to_print("Start....")

    # --------------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    args.home = str(Path.home())
    args.data_path = os.path.join(args.home, args.data_path)

    # --------------------------------------------------------------------------------------------------------------
    # --- load data ----
    to_print("Data loading....")

    keys = np.load(os.path.join(args.data_path, "sky_keys.npy"))
    #sources_inpixels = np.load(os.path.join(args.data_path, "sky_sources_px.npy"), allow_pickle=True)
    sources = np.load(os.path.join(args.data_path, "sky_sources.npy"), allow_pickle=True)

    # ------------------------
    TP = []
    FN = []
    FP = []
    T = []

    for i in range(args.n):

        key = keys[i]
        src = sources[key]

        # --- source detection casa ----
        headers_info = os.path.join(args.data_path, "fits", "dirty_gaussians_" + key + ".fits")
        header = prepareHeaders(fits.getheader(headers_info))

        input_path = os.path.join("path_wher_to_save", "name.fits")
        res_dir = "path_wher_to_save"
        try:
            fits.writeto(res_dir, image, header, overwrite=True)
            # --- process created fits image ---
            applyPyBDSF(input_path, res_dir, args, i, is_debug=False)

        except RuntimeError:
            image += np.random.normal(0, args.sigma, image.shape)
            fits.writeto(input_path, image, header, overwrite=True)
            # --- process created fits image ---
            try:
                applyPyBDSF(input_path, res_dir, args, is_debug=True)
            except RuntimeError:
                print("\n\n *************** RuntimeError: {i={i}} **********\n\n ")
                continue

        if os.path.exists(os.path.join(res_dir, "sources.cat")):
            t = Table.read(os.path.join(res_dir, "sources.cat"), format='ascii')
            # -----------------------------------------------------------
            Z = np.zeros(len(src))
            tp = 0
            fp = 0
            for row in range(len(t)):
                x = t[row]['col3']
                y = t[row]['col5']

                dists = np.zeros((len(src)))
                for i1 in range(len(src)):
                    dists[i1] = math.sqrt((x - src[i1][0]) ** 2 + (y - src[i1][1]) ** 2)

                min_ind = np.argmin(dists)
                if dists[min_ind] <= args.d:
                    Z[min_ind] = 1
                else:
                    fp += 1

            tp = np.sum(Z)
            TP.append(tp)
            FP.append(fp)  # fa
            FN.append(len(src) - tp)  # miss
            T.append(len(src))

            to_print(f"T = {len(src)}\t TP = {tp}\t FN = {len(src) - tp}\t FP = {fp}\n")
            os.remove(os.path.join(res_dir, "sources.cat"))

        else:
            TP.append(0)
            FP.append(0)  # fa
            FN.append(len(src))  # miss
            T.append(len(src))

            to_print(f"T = {len(src)}\t TP = 0\t FN = {len(src)}\t FP = 0\n")

    TP = np.sum(np.asarray(TP))
    FN = np.sum(np.asarray(FN))
    FP = np.sum(np.asarray(FP))
    T = np.sum(np.asarray(T))

    purity = TP / (TP + FP)
    completeness = TP / (TP + FN)

    to_print(f"\nT = {T}\t TP = {TP}\t FN = {FN}\t FP = {FP}")
    to_print(f"purity = {purity}\t "
             f"completeness = {completeness}\n")

    to_print("Test is finished.")