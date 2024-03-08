import functools
from typing import Tuple, Union, List, Callable, Optional

import pandas as pd
import tifffile as tiff
import torch
from torch.utils.data import Dataset as PyTorchDataset


def load_image_from_df_cached(
    df: pd.DataFrame,
    max_cache_size: int,
) -> Callable:
    """ Returns a function that loads an image from a dataframe, wrapped in an lru cache. """
    def f(index: int) -> torch.Tensor:
        image_fpath = df['filepath'].iloc[index]
        im = tiff.imread(image_fpath)
        im = torch.tensor(im)
        return im
    return functools.lru_cache(maxsize=max_cache_size)(f)


class CompoundDataset(PyTorchDataset):
    def __init__(
            self,
            csv_fpath: str,
            max_cache_size: Optional[int] = 0,
    ):
        df = pd.read_csv(csv_fpath)
        if 'treatment' not in df.columns:
            df['treatment'] = df['compound_name'].astype(str) + '_' + df['compound_uM'].astype(str)
        self.df = df
        self.load_im = load_image_from_df_cached(df=self.df, max_cache_size=max_cache_size)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.load_im(index), index

    def __len__(self) -> int:
        return len(self.df)

    def get_df(self) -> pd.DataFrame:
        return self.df


class IndexFilteredDataset(PyTorchDataset):
    def __init__(self, source_dataset: CompoundDataset, retained_indices: List[int]):
        super().__init__()
        self.source_dataset = source_dataset
        self.retained_indices = retained_indices

    def __getitem__(self, subset_index: int) -> Tuple[torch.Tensor, int]:
        im, _ = self.source_dataset[int(self.retained_indices[subset_index])]
        return im, subset_index

    def __len__(self) -> int:
        return len(self.retained_indices)

    def get_df(self) -> pd.DataFrame:
        return self.source_dataset.get_df().iloc[self.retained_indices].reset_index(drop=True)


