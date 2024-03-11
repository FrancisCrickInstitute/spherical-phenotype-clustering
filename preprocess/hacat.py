"""
HaCaT cells with JUMP MOA library, tiled 4x4.

This script has hardcoded paths, but we include it here for reference.
"""
import os
import sys
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import exposure
import tifffile as tiff
import torch
from torchvision.transforms import functional as tv_F


DATASET_NAME = '20230515_prosperity_jump_moa_hacat'

PROJECT_FPATH = os.path.join(
    "/nemo/project/proj-prosperity/hts/raw/projects/",
    DATASET_NAME,
)

PLATES_FOLDER = os.path.join(
    PROJECT_FPATH,
    'data/raw_data/phenix/',
    'Joe_15052023_HaCaT 40x JUMP triplicate run',
)

PLATE_DIRS = [
    '30544__2023-05-13T20_20_02-Measurement 1',
    '30545__2023-05-13T22_11_36-Measurement 1',
    '30546__2023-05-14T00_03_20-Measurement 1',
    '30547__2023-05-14T01_53_56-Measurement 1',
    '30548__2023-05-14T03_45_58-Measurement 1',
    '30549__2023-05-14T05_38_54-Measurement 1',
    '30550__2023-05-14T07_29_28-Measurement 1',
    '30551__2023-05-14T09_22_45-Measurement 1',
    '30552__2023-05-14T11_14_51-Measurement 1',
]

PLATEMAP_FPATH = os.path.join(
    PROJECT_FPATH, 'Compound platemap', 'GSKised JUMP reference set plate map.xlsx',
)

PLATE_DICT = {
    30544: {'replicate': 1, 'compound_uM': 0.1, 'cell_type': 'HaCaT iCas9 WT'},
    30545: {'replicate': 2, 'compound_uM': 0.1, 'cell_type': 'HaCaT iCas9 WT'},
    30546: {'replicate': 3, 'compound_uM': 0.1, 'cell_type': 'HaCaT iCas9 WT'},
    30547: {'replicate': 1, 'compound_uM': 1, 'cell_type': 'HaCaT iCas9 WT'},
    30548: {'replicate': 2, 'compound_uM': 1, 'cell_type': 'HaCaT iCas9 WT'},
    30549: {'replicate': 3, 'compound_uM': 1, 'cell_type': 'HaCaT iCas9 WT'},
    30550: {'replicate': 1, 'compound_uM': 10, 'cell_type': 'HaCaT iCas9 WT'},
    30551: {'replicate': 2, 'compound_uM': 10, 'cell_type': 'HaCaT iCas9 WT'},
    30552: {'replicate': 3, 'compound_uM': 10, 'cell_type': 'HaCaT iCas9 WT'},
}

CHANNEL_NAMES = [
    "HOECHST 33342",
    "Alexa 488",
    "MitoTracker Deep Red",
    "Alexa 568",
]

DNA_CHANNEL = CHANNEL_NAMES[0]

SAVE_FOLDER = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'data', 'hacat',
)
UNTILED_CSV_FPATH = os.path.join(SAVE_FOLDER, f'{DATASET_NAME}.csv')

OUT_DATASET_NAME = "tiled4x4__relative_clip__" + DATASET_NAME
DATA_FOLDER = "/nemo/project/proj-prosperity/MLAnalysis/nightil/data/"
TILE_SAVE_FOLDER = os.path.join(DATA_FOLDER, 'HacatTiled', OUT_DATASET_NAME)
OUT_TILED_CSV_FPATH = os.path.join(DATA_FOLDER, 'HacatTiled', OUT_DATASET_NAME + '.csv')
if not os.path.exists(TILE_SAVE_FOLDER):
    os.makedirs(TILE_SAVE_FOLDER)


def get_channel_fpath(df, channel_name):
    return df[df['Channel Name'] == channel_name].iloc[0]['URL']\
        .replace('/camp/stp/hts/working/scott/projects', '/nemo/project/proj-prosperity/hts/raw/projects')\
        .replace('/nemo/stp/hts/working/scott/projects', '/nemo/project/proj-prosperity/hts/raw/projects')


def get_intensity_percentiles(
        plate_folder: str,
        plate_dirs: List[str],
        dna_channel_name: str,
        p_lo: int = 0.01,
        p_hi: int = 99.9
):
    intensities = np.zeros(1000000, dtype=np.int64)

    print("Computing percentiles...")
    for p, plate_dir in enumerate(plate_dirs):
        index_df = pd.read_csv(os.path.join(plate_folder, plate_dir, "indexfile.txt"), delimiter="\t")
        groups = index_df.groupby(['Row', 'Column', 'Field']).groups

        for row, col, field in tqdm(groups,
                                    desc=f"Plate {p + 1}/{len(plate_dirs)}: {os.path.basename(plate_dir).split('__')[0]}"):
            group = index_df.iloc[groups[(row, col, field)]]

            impath = get_channel_fpath(group, dna_channel_name)

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


def dna_foreground_pct(dna_channel, dna_lo, dna_hi):
    dna_channel = exposure.rescale_intensity(
        dna_channel, in_range=(dna_lo, dna_hi), out_range=(0, 1)
    )

    th_01 = 0.1
    foreground_pct = 100.*np.count_nonzero(dna_channel > th_01)/(dna_channel.shape[0]*dna_channel.shape[1])

    return foreground_pct


def make_hts_dataset_csv(
        plate_folder: str,
        plate_dirs: List[str],
        platemap_df: pd.DataFrame,
        plate_dict: dict,
        channel_names: List[str],
        dna_channel_name: str,
        dna_lo: int,
        dna_hi: int,
        plate_to_platemap: dict = None,
):
    """
    platemap_df info used:
        Well,
        Row,
        Column,
        Regno (compound id, indexes everything else),
        Compound name,
        MoA
    """
    df = pd.DataFrame(
        columns=[
                    'plate',
                    'dir',
                    'table_nr',
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
                    'i_lo',
                    'i_hi',
                ] + channel_names
    )

    # plate folder format seems to be along the lines of: "30721__2023-06-15T19_38_04-Measurement 1"
    print()
    print("Enumerating plate directories to compose dataframe...")
    for p, plate_dir in enumerate(plate_dirs):
        barcode = int(plate_dir.split("__")[0])

        if plate_to_platemap is not None:
            platemap_df = plate_to_platemap[barcode]

        compound_dict = {
            'DMSO': {'moa': 'DMSO', 'name': 'DMSO'}
        }
        for _, row in platemap_df.iterrows():
            if row['Compound name'] != 'DMSO':
                compound_dict[row['Regno']] = {
                    'moa': row['MoA'].strip(),
                    'name': str(row['Compound name']).strip(),
                }

        replicate = plate_dict[barcode]['replicate']
        compound_uM = plate_dict[barcode]['compound_uM']
        cell_type = plate_dict[barcode]['cell_type']

        index_df = pd.read_csv(os.path.join(plate_folder, plate_dir, "indexfile.txt"), delimiter="\t")
        groups = index_df.groupby(['Row', 'Column', 'Field']).groups

        for row, col, field in tqdm(groups, desc=f"Plate {p + 1}/{len(plate_dirs)}: {barcode}"):
            well_info = platemap_df[(platemap_df['Row'] == row) & (platemap_df['Column'] == col)].iloc[0]

            if well_info['Compound name'] == 'DMSO':
                compound_id = 'DMSO'
            else:
                compound_id = well_info['Regno']

            group = index_df.iloc[groups[(row, col, field)]]
            dna_fpath = get_channel_fpath(group, dna_channel_name)
            dna_channel = imread(dna_fpath)
            cell_pct = dna_foreground_pct(dna_channel, dna_lo, dna_hi)

            df_row = {
                'plate': barcode,
                'dir': plate_dir,
                'table_nr': 1,
                'compound_id': compound_id,
                'compound_name': compound_dict[compound_id]['name'],
                'compound_uM': compound_uM,
                'moa': compound_dict[compound_id]['moa'],
                'well': well_info['Well'],
                'replicate': replicate,
                'field': field,
                'image_nr': 1,  # With max projection image number is N/A
                'cell_type': cell_type,
                'cell_pct': cell_pct,
                'i_lo': dna_lo,
                'i_hi': dna_hi,
            }
            for channel_name in channel_names:
                df_row[channel_name] = get_channel_fpath(group, channel_name)

            df = pd.concat([df, pd.DataFrame(df_row, index=[0])], ignore_index=True)

    return df


def load_and_preprocess_image(filename_channel1, filename_channel2, filename_channel3, filename_channel4):
    channel_dapi = imread(filename_channel1)
    channel_alexa_488 = imread(filename_channel2)
    channel_mito = imread(filename_channel3)
    channel_alexa_568 = imread(filename_channel4)

    im = np.stack([channel_dapi, channel_alexa_488, channel_mito, channel_alexa_568], axis=0)
    im = im.astype(np.float32)
    im = torch.tensor(im)
    im = tv_F.resize(im, [512, 512], interpolation=tv_F.InterpolationMode.BICUBIC)
    im = im.numpy()

    # per channel clipping and normalization
    for c in range(4):
        lo, hi = np.percentile(im[c, :, :], 0.01), np.percentile(im[c, :, :], 99.9)
        im[c, :, :] = np.clip(im[c, :, :], lo, hi)
        im[c, :, :] = (im[c, :, :] - lo)/(hi - lo)

    im = im*255
    im = im.astype(np.uint8)

    return im


def generate_tiles(im, dna_threshold, tile_size=128):
    assert im.shape[1] % tile_size == 0
    assert im.shape[2] % tile_size == 0

    n_tiles_i = im.shape[1]//tile_size
    n_tiles_j = im.shape[2]//tile_size

    tiles = {}
    tile_number = 1
    for i in range(n_tiles_i):
        for j in range(n_tiles_j):
            tile = im[:, i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
            dna_foreground = np.count_nonzero(tile[0, :, :] > dna_threshold)
            dna_foreground_pct = 100.*dna_foreground/(tile_size*tile_size)
            tiles[tile_number] = (tile, dna_foreground_pct)
            tile_number += 1
    return tiles


def main():
    platemap_df = pd.read_excel(PLATEMAP_FPATH)

    lo, hi = get_intensity_percentiles(
        plate_folder=PLATES_FOLDER,
        plate_dirs=PLATE_DIRS,
        dna_channel_name=DNA_CHANNEL
    )

    untiled_dataset_df = make_hts_dataset_csv(
        plate_folder=PLATES_FOLDER,
        plate_dirs=PLATE_DIRS,
        platemap_df=platemap_df,
        plate_dict=PLATE_DICT,
        channel_names=CHANNEL_NAMES,
        dna_channel_name=DNA_CHANNEL,
        dna_lo=lo,
        dna_hi=hi,
    )

    print()
    print("Saving dataframe...", end='')
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    untiled_dataset_df.to_csv(UNTILED_CSV_FPATH, index=False)
    print("DONE")

    out_tile_df = pd.DataFrame(
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
            'cell_type',
            'cell_pct',
            'tile',
            'otsu_threshold',
        ]
    )

    for _, row in tqdm(untiled_dataset_df.iterrows(), total=len(untiled_dataset_df)):
        # don't include low content images
        if row['cell_pct'] >= 1:
            uint8_im = load_and_preprocess_image(
                row["HOECHST 33342"],
                row["Alexa 488"],
                row["MitoTracker Deep Red"],
                row["Alexa 568"],
            )

            nuclei_th = threshold_otsu(uint8_im[0, :, :])
            tiles = generate_tiles(uint8_im, nuclei_th, tile_size=128)

            for tile_number, (tile, tile_dna_foreground_pct) in tiles.items():
                # save the tile
                tile_save_fpath = os.path.join(TILE_SAVE_FOLDER, f"{row['plate']}_{row['well']}_{row['field']}_{tile_number}.tiff")
                tiff.imwrite(tile_save_fpath, tile)

                # add a new row to the output dataframe
                out_tile_df = pd.concat([out_tile_df, pd.DataFrame({
                    'plate': row['plate'],
                    'filepath': tile_save_fpath,
                    'compound_id': row['compound_id'],
                    'compound_name': row['compound_name'],
                    'compound_uM': row['compound_uM'],
                    'moa': row['moa'],
                    'well': row['well'],
                    'replicate': row['replicate'],
                    'field': row['field'],
                    'cell_type': row['cell_type'],
                    'cell_pct': tile_dna_foreground_pct,
                    'tile': tile_number,
                    'otsu_threshold': nuclei_th,
                }, index=[0])], ignore_index=True)

    # save the dataframe
    out_tile_df.to_csv(OUT_TILED_CSV_FPATH, index=False)


if __name__ == '__main__':
    main()


