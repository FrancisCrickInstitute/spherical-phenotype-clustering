"""
This script processes images from the BBBC021 dataset according to the pipeline described in Ando et al. (2017).
Dataset images and index files downloaded from the link: https://bbbc.broadinstitute.org/BBBC021
The script requires:
    - images and medata stored locally in folder: root_folder (hardcoded)
    - csv filenames and folder paths for storing output
The script generates:
    - folders containing mean channel images, illumination correction functions and single-cell crops
    - csv files specifying location of image crops (to be feed to the clustering method)
"""

import os
import shutil
import numpy as np
import pandas as pd
import torch

from collections import OrderedDict
from tqdm import tqdm
from typing import List, Union, Hashable
from tifffile import imread, imwrite
from scipy.ndimage import percentile_filter, gaussian_filter
from skimage import exposure
from skimage.filters import threshold_otsu


def ignore_files(_dir, files):
    return [f for f in files if os.path.isfile(os.path.join(_dir, f))]


def dna_foreground_pct(dna_channel, dna_lo, dna_hi):
    dna_channel = exposure.rescale_intensity(
        dna_channel, in_range=(dna_lo, dna_hi), out_range=(0, 1)
    )
    th_01 = 0.1
    foreground_pct = 100.*np.count_nonzero(dna_channel > th_01)/(dna_channel.shape[0]*dna_channel.shape[1])

    return foreground_pct


def get_intensity_percentiles(csv_fpath: str, p_dirs: List[str], metadata, p_lo=0.01, p_hi=99.9):
    intensities = np.zeros(1000000, dtype=np.int64)

    print("Computing percentiles...")
    for q, p_dir in enumerate(p_dirs):
        p_name = str(p_dir.split("/")[0])
        idx_df = metadata[metadata.plate == p_name]

        for _idx, _row in tqdm(idx_df.iterrows(),
                               desc=f"Plate {q+1}/{len(p_dirs)}: {p_name}"):
            impath = os.path.join(csv_fpath,
                                  _row['Image_PathName_DAPI'].split('/')[1],
                                  _row['Image_FileName_DAPI'])
            im = imread(impath)
            if im.max() > len(intensities):
                _intensities = intensities
                intensities = np.zeros(im.max() + 1000)
                intensities[:len(_intensities)] = _intensities[:]
            uniq, counts = np.unique(im.ravel(), return_counts=True)
            intensities[uniq] += counts

    total_count = np.sum(intensities)
    percents = 100. * np.cumsum(intensities) / total_count

    lo = (percents > p_lo).nonzero()[0][0]
    hi = (percents > p_hi).nonzero()[0][0]

    print(f"Finished computing percentiles... lo@{p_lo}={lo}, hi@{p_hi}={hi}")
    return lo, hi


def get_bbbc021_metadata(csv_fpath: str):
    """
    Function loading the BBBC021 dataset.
    It requires local copies of metadata files for the dataset.
    Download from: https://bbbc.broadinstitute.org/BBBC021

    Args:
        csv_fpath: path to metadata files

    Returns:
        Tensor: mean of all images belonging to specified group
        Tensor: flatfield image for specified group
    """
    try:
        metadata = pd.read_csv(os.path.join(csv_fpath, 'BBBC021_v1_image.csv'))
        compound = pd.read_csv(os.path.join(csv_fpath, 'BBBC021_v1_moa.csv'))
        cell_locations = pd.read_csv(os.path.join(csv_fpath, 'BBBC021_v1_object.txt'), delimiter='\t', header=None)
        cell_locations = cell_locations.rename(columns={0: 'TableNumber',
                                                        1: 'ImageNumber',
                                                        2: 'ObjectNumber',
                                                        3: 'Nuclei_Location_Center_X',
                                                        4: 'Nuclei_Location_Center_Y',
                                                        198: 'Cells_Location_Center_X',
                                                        199: 'Cells_Location_Center_Y',
                                                        }
                                               )
    except OSError:
        raise IOError("Missing metadata files \n")
    metadata = metadata.rename(columns={
        'Image_Metadata_Compound': 'compound_name',
        'Image_Metadata_Concentration': 'compound_uM',
        'Image_Metadata_Plate_DAPI': 'plate'
    })
    compound = compound.reset_index().rename(columns={
        'index': 'compound_id',
        'compound': 'compound_name',
        'concentration': 'compound_uM'
    })
    metadata = metadata.merge(compound, on=['compound_name', 'compound_uM'], how='left')
    metadata['moa'].fillna('unknown', inplace=True)
    return metadata, cell_locations


def make_illumination_correction_images(df_group: pd.DataFrame,
                                        group: Union[str, Hashable],
                                        channel_dict: OrderedDict,
                                        illumination_mean_folder: str,
                                        icf_folder: str,
                                        filter_pctl: int = 10,
                                        filter_size: int = 200,
                                        filter_sigma: int = 50,
                                        ):
    """
    Function implementing the method for correcting illumination.
    Adapted from Singh et al. (2014).

    Args:
        df_group: collection of images to be used to calculate illumination correction function
        group: group barcode
        channel_dict: channels specification
        illumination_mean_folder: storage folder for group channels means
        icf_folder: storage folder for group illumination correction functions

    Returns:
        Tensor: mean of all images belonging to specified group
        Tensor: flatfield image for specified group
    """
    assert filter_size % 2 == 0
    ch_means = []
    try:  # check if mean illumination mask already exists
        im_mean = imread(f"{illumination_mean_folder}/{group}_illumination_mean.tif")
        ch_means = torch.unbind(torch.tensor(im_mean))
        ch_means = [ch_img.numpy() for ch_img in ch_means]
    except IOError:
        # loop over the images and add them
        N = len(df_group)
        for i, row in tqdm(df_group.reset_index(drop=True).iterrows(), total=N, desc=f"Group {group}"):
            for ch, (key, name) in enumerate(channel_dict.items()):
                # create the aggregate images for each channel
                ch_fpath = row[name]
                ch_img = imread(ch_fpath)
                if i == 0:
                    im_shape = np.shape(ch_img)
                    ch_means.append(np.zeros(im_shape, dtype=np.float32))
                assert (np.shape(ch_img)[0] == im_shape[0]) & (np.shape(ch_img)[1] == im_shape[1])
                ch_means[ch] += ch_img / N

        im_mean = np.stack(ch_means, axis=0).copy()
        imwrite(f"{illumination_mean_folder}/{group}_illumination_mean.tif", im_mean)

    # apply the filter kernel sizes
    try:
        im_correction = imread(f"{icf_folder}/{group}_illumination_correction_{filter_size}.tif")
    except IOError:
        ch_filtered = []
        for ch, _ in enumerate(channel_dict.items()):
            ch_filtered.append(
                gaussian_filter(percentile_filter(ch_means[ch], filter_pctl, size=filter_size+1), sigma=filter_sigma))
        im_correction = np.stack(ch_filtered, axis=0)
        # save the correction image
        imwrite(f"{icf_folder}/{group}_illumination_correction_{filter_size}.tif", im_correction)

    return im_mean, im_correction


def load_and_preprocess_image(csv_fpaths: List[str],
                              dna_lo,
                              dna_hi,
                              dna_channel_nr,
                              preprocessing_scheme: str,
                              flatfield_im: np.ndarray,
                              a_min: float = 1,
                              a_max: float = 5,
                              ):
    """
    Function to load and preprocess images.
    Different preprocessing methods implemented (see manuscript)

    Args:
        csv_path: locations of images for each channel
        dna_lo, dna_hi: dna intensity thresholds for clipping (used in some preprocessing schemes)
        dna_channel_nr: index of dna channel
        preprocessing_scheme: barcode specifying preprocessing scheme
        flatfield_im: generated flatfield img for illumination correction

    Returns:
        Tuple[Tensor, float]: preprocessed image and foreground percentage in DNA channel.
    """
    channel_stack = []
    for fpath in csv_fpaths:
        ch = imread(fpath)
        ch = ch.astype(np.float32)
        channel_stack.append(ch)

    pct = dna_foreground_pct(channel_stack[dna_channel_nr], dna_lo, dna_hi)

    im = np.stack(channel_stack, axis=0)
    # set flatfield image to 1 if not otherwise specified
    if flatfield_im.size == 0:
        flatfield_im = np.ones_like(im)
    # divide by flatfield image
    im = np.divide(im, flatfield_im)

    # implement different preprocessing schemes
    if preprocessing_scheme == 'raw':
        lo = np.min(im)
        hi = np.max(im)
        im = (im - lo) / (hi - lo)
    elif preprocessing_scheme == 'default':
        lo = np.percentile(im, 0.01)
        hi = np.percentile(im, 99.9)
        im = np.clip(im, lo, hi)
        im = (im - lo) / (hi - lo)
    elif preprocessing_scheme == 'relative_clip':
        for ch in range(0, len(im)):
            lo = np.percentile(im[ch], 0.01)
            hi = np.percentile(im[ch], 99.9)
            im[ch] = np.clip(im[ch], lo, hi)
            im[ch] = (im[ch] - lo) / (hi - lo)
    elif preprocessing_scheme == 'ando':
        lo = np.min(im)
        im = a_min + (im - lo)  # set minimum to 1
        im = np.log(im)  # take log
        im = np.clip(im, a_min=None, a_max=a_max)  # clipping
        im = im / a_max
    elif preprocessing_scheme == 'relative_ando':
        for ch in range(0, len(im)):
            lo = np.min(im[ch])
            im[ch] = a_min + (im[ch] - lo)  # set minimum to 1
            im[ch] = np.log(im[ch])  # take log
            im[ch] = np.clip(im[ch], a_min=None, a_max=a_max)  # clipping
            im[ch] = im[ch] / a_max
    elif preprocessing_scheme == 'relative_log':
        for ch in range(0, len(im)):
            lo = np.min(im[ch])
            im[ch] = a_min + (im[ch] - lo)  # set minimum to 1
            im[ch] = np.log(im[ch])  # take log
            lo = np.percentile(im[ch], 0.01)
            hi = np.percentile(im[ch], 99.9)
            im[ch] = np.clip(im[ch], a_min=lo, a_max=hi)  # clipping to percentiles
            im[ch] = (im[ch] - lo) / (hi - lo)  # rescale
    else:
        raise NotImplementedError('Preprocessing scheme not implemented. \n')

    im = im * 255
    im = im.astype(np.uint8)

    return im, pct


def generate_crops(im,
                   centroids,
                   dna_threshold,
                   dna_channel_nr: int = 0,
                   crop_size: int = 96,
                   extrapolation_method: str = None,
                   extrapolation_value: str = None
                   ):
    """
    Function to generate single-cell crops.
    Two methods implemented for handling crops of cells near image edges:
        1. extrapolate (pixels outside the image set to extrapolation value)
        2. shift (crop shifted to put it inside the image)
    If method not specified, cells close to edges are discarded.

    Args:
        im (Tensor): image to be cropped.
        centroids: list of cell nuclei positions in the form List[cell_id, nucleus center_x, nucleus center_y]
        dna_threshold: intensity threshold to filter out mostly empty crops

    Returns:
        List[Tuple[Tensor, float]]: List of tuples containing crops and foreground percentage.
    """
    assert crop_size % 2 == 0
    c, h, w = np.shape(im)
    crop_halfdim = int(0.5 * crop_size)
    crops = {}
    for crop_number, center_x, center_y in centroids:
        x = int(center_x)
        y = int(center_y)
        crop = np.zeros((c, crop_size, crop_size), dtype=im.dtype)
        # set crop cropping indices
        c_ymin = 0
        c_ymax = crop_size
        c_xmin = 0
        c_xmax = crop_size
        # set img cropping indices
        x_min = x-crop_halfdim
        x_max = x+crop_halfdim
        y_min = y-crop_halfdim
        y_max = y+crop_halfdim
        if (x-crop_halfdim >= 0) & (x+crop_halfdim <= w) & \
                (y-crop_halfdim >= 0) & (y+crop_halfdim <= h):
            pass
        else:
            if extrapolation_method is None:
                continue
            elif extrapolation_method == 'extrapolate':
                assert extrapolation_value is not None
                if x-crop_halfdim < 0:
                    x_min = 0
                    x_max = x+crop_halfdim
                    c_xmin = crop_halfdim-x
                    c_xmax = crop_size
                elif x+crop_halfdim > w:
                    x_min = x-crop_halfdim
                    x_max = w
                    c_xmin = 0
                    c_xmax = crop_halfdim+w-x
                else:
                    pass
                if y-crop_halfdim < 0:
                    y_min = 0
                    y_max = y+crop_halfdim
                    c_ymin = crop_halfdim-y
                    c_ymax = crop_size
                elif y+crop_halfdim > h:
                    y_min = y-crop_halfdim
                    y_max = h
                    c_ymin = 0
                    c_ymax = crop_halfdim+h-y
                else:
                    pass
            elif extrapolation_method == 'shift':
                if x-crop_halfdim < 0:
                    x_min = 0
                    x_max = crop_size
                elif x+crop_halfdim > w:
                    x_min = -1*crop_size
                    x_max = w
                else:
                    pass
                if y-crop_halfdim < 0:
                    y_min = 0
                    y_max = crop_size
                elif y+crop_halfdim > h:
                    y_min = -1*crop_size
                    y_max = h
                else:
                    pass
            else:
                raise NotImplementedError(f"{extrapolation_method} method for edge treatment not implemented.")
        crop[:, c_ymin:c_ymax, c_xmin:c_xmax] = im[:, y_min:y_max, x_min:x_max]
        foreground = np.count_nonzero(crop[dna_channel_nr, :, :] > dna_threshold)
        foreground_pct = 100. * foreground / (crop_size * crop_size)
        crops[int(crop_number)] = (crop, foreground_pct)

    return crops


def make_bbbc021_crops_csv(
        plate_dirs: List[str],
        dataset_fpath: str,
        processed_dataset_fpath: str,
        metadata: pd.DataFrame,
        cell_locations: pd.DataFrame,
        channel_dict: OrderedDict,
        dna_lo: int,
        dna_hi: int,
        crop_size: int = 96,
        dna_channel_nr: int = 0,
        get_illumination_correction: bool = False,
        preprocessing_scheme: str = 'default',
        a_min: int = 1,
        a_max: int = 5,
        filter_pctl: int = 10,
        filter_size: int = 200,
        filter_sigma: int = 50,
        extrapolation_method: str = None,
        extrapolation_value: int = None
):
    """
    Function running the preprocessing pipeline.

    Args:
            plate_dirs:
            dataset_fpath: path to image storage folder
            processed_dataset_fpath: path to crops
            metadata: df containing image metadata
            cell_locations: df containing cell nuclei locations
            channel_dict: channels specification
            dna_lo, dna_hi: thresholds on dna intensity for preprocessing

    Returns:
        pd.DataFrame: summary table for single-cell crops (incl. image locations)
    """
    _df = pd.DataFrame(
        columns=[
            'plate',
            'filepath',
            'compound_id',
            'compound_name',
            'compound_uM',
            'moa',
            'well',
            'replicate',
            'field',
            'image_nr',
            'cell_type',
            'cell_pct',
            'tile',
            'otsu_threshold',
        ]
    )
    if get_illumination_correction:
        illumination_mean_save_folder = os.path.join(
            os.path.dirname(processed_dataset_fpath),
            'illumination_mean_groupby_plate__BBBC021'
        )
        if not os.path.exists(illumination_mean_save_folder):
            os.makedirs(illumination_mean_save_folder)
        icf_save_folder = os.path.join(
            os.path.dirname(processed_dataset_fpath),
            'icf_groupby_plate_fs' + str(filter_size) + '__BBBC021'
        )
        if not os.path.exists(icf_save_folder):
            os.makedirs(icf_save_folder)

    icf_image = np.array([])

    print()
    print("Enumerating plate directories to compose dataframe and preprocess images...")
    for p, plate_dir in enumerate(plate_dirs):

        barcode = str(plate_dir.split("/")[0])
        cell_type = 'Human MCF7'
        index_df = metadata.query("plate == @barcode")
        for key, name in channel_dict.items():
            index_df[name] = index_df[['Image_PathName_' + key, 'Image_FileName_' + key]].apply(
                lambda x: os.path.join(dataset_fpath,
                                       x['Image_PathName_' + key].split('/')[1],
                                       x['Image_FileName_' + key]
                                       ), axis=1
            )

        # perform the illumination correction
        if get_illumination_correction:
            _, icf_image = make_illumination_correction_images(index_df,
                                                               barcode,
                                                               channel_dict,
                                                               illumination_mean_save_folder,
                                                               icf_save_folder,
                                                               filter_pctl=filter_pctl,
                                                               filter_size=filter_size,
                                                               filter_sigma=filter_sigma,
                                                               )

        for idx, row in tqdm(index_df.iterrows(),
                             desc=f"Plate {p+1}/{len(plate_dirs)}: {barcode}"):

            # set the metadata
            image_nr = row['ImageNumber']
            well_info = row['Image_Metadata_Well_DAPI']
            field = row['Image_FileName_DAPI'].split('_')[-2]
            replicate = row['Replicate']
            compound_id = row['compound_id']
            compound_name = row['compound_name']
            compound_uM = row['compound_uM']
            table_nr = row['TableNumber']
            moa = row['moa']

            # preprocess the raw images
            channel_fpaths = [row[name] for name in channel_dict.values()]
            img, cell_pct = load_and_preprocess_image(channel_fpaths,
                                                      dna_lo,
                                                      dna_hi,
                                                      dna_channel_nr,
                                                      preprocessing_scheme,
                                                      icf_image,
                                                      a_min=a_min,
                                                      a_max=a_max
                                                      )

            # generate single cell crops
            cell_nuclei = cell_locations.loc[
                (cell_locations.TableNumber == table_nr) & (cell_locations.ImageNumber == image_nr),
                ['ObjectNumber', 'Cells_Location_Center_X', 'Cells_Location_Center_Y']
            ].to_numpy()
            nuclei_th = threshold_otsu(img[2, :, :])
            crops = generate_crops(img,
                                   cell_nuclei,
                                   nuclei_th,
                                   dna_channel_nr,
                                   crop_size,
                                   extrapolation_method,
                                   extrapolation_value
                                   )

            for crop_number, (crop, crop_cell_pct) in crops.items():
                # save the crop
                crop_save_fpath = os.path.join(processed_dataset_fpath,
                                               row['Image_PathName_DAPI'].split('/')[1],
                                               f"{barcode}_{well_info}_{field}_{table_nr}_{image_nr}_{crop_number}.tif"
                                               )
                imwrite(crop_save_fpath, crop)

                # compose dataframe
                _df = pd.concat([_df, pd.DataFrame({
                    'plate': barcode,
                    'filepath': crop_save_fpath,
                    'table_nr': table_nr,
                    'compound_id': compound_id,
                    'compound_name': compound_name,
                    'compound_uM': compound_uM,
                    'moa': moa,
                    'well': well_info,
                    'replicate': replicate,
                    'field': field,
                    'image_nr': image_nr,
                    'cell_type': cell_type,
                    'cell_pct': cell_pct,
                    'tile': crop_number,
                    'otsu_threshold': nuclei_th,
                }, index=[0])], ignore_index=True)

    return _df


"""
Main instructions
"""
# setting up input/output folders
root_folder = '/nemo/stp/ddt/working/cairoli/prosp-phen/prosp-phen/data/BBBC021'
if not os.path.exists(root_folder):
    os.makedirs(root_folder)
images_folder = '/nemo/project/proj-prosperity/MLAnalysis/nightil/cairoli/data/BBBC021'
if not os.path.exists(images_folder):
    os.makedirs(images_folder)

# getting the metadata
metadata_df, cell_locations_df = get_bbbc021_metadata(root_folder)
plate_dirs = [plate_dir + '/' for plate_dir in metadata_df['plate'].unique()]
channel_dict = OrderedDict([
    ('Tubulin', 'filename_tubulin'),
    ('Actin', 'filename_actin'),
    ('DAPI', 'filename_dapi')
])

# setting storage folders for cropped images
bbbc021__raw_folder = os.path.join(root_folder, 'data_downloaded')

# folders for single-cell crops with preprocessing
bbbc021__cropped96__relative_ando__icf_groupby_plate_fs200__folder = os.path.join(
    images_folder,
    'BBBC021__cropped96__relative_ando__icf_groupby_plate_fs200'
)
if not os.path.exists(bbbc021__cropped96__relative_ando__icf_groupby_plate_fs200__folder):
    shutil.copytree(bbbc021__raw_folder,
                    bbbc021__cropped96__relative_ando__icf_groupby_plate_fs200__folder,
                    ignore=ignore_files
                    )

print()
print("Get intensity percentiles on DNA channel of raw images...")
LO, HI = get_intensity_percentiles(bbbc021__raw_folder, plate_dirs, metadata_df)


print()
print("Generating single-cell crops with relative ando preprocessing and illumination correction (filter size=200)...")
bbbc021__cropped96__relative_ando__icf_groupby_plate_fs200__csvfile = make_bbbc021_crops_csv(
    plate_dirs,
    bbbc021__raw_folder,
    bbbc021__cropped96__relative_ando__icf_groupby_plate_fs200__folder,
    metadata_df,
    cell_locations_df,
    channel_dict,
    LO, HI,
    dna_channel_nr=2,
    crop_size=96,
    get_illumination_correction=True,
    preprocessing_scheme='relative_ando',
)
print("Finishing creating dataframe")

print()
print("Saving dataframe...", end='')
bbbc021__cropped96__relative_ando__icf_groupby_plate_fs200__csvfile.to_csv(
    os.path.join(root_folder, 'BBBC021__cropped96__relative_ando__icf_groupby_plate_fs200.csv')
)
bbbc021_wcontrol__cropped96__relative_ando__icf_groupby_plate_fs200__csvfile = \
    bbbc021__cropped96__relative_ando__icf_groupby_plate_fs200__csvfile.copy().query("moa != 'unknown'")  # remove unknowns
bbbc021_wcontrol__cropped96__relative_ando__icf_groupby_plate_fs200__csvfile.to_csv(
    os.path.join(root_folder, f'BBBC021_trim_with_control__cropped96__relative_ando__icf_groupby_plate_fs200.csv')
)
print("DONE")


