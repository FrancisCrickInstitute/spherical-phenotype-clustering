from typing import Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from dataset import CompoundDataset, IndexFilteredDataset


class Embedder(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_out_dim: int, embed_dim: int):
        super(Embedder, self).__init__()
        self.encoder = encoder
        self.mlp_head = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim),
            nn.ReLU(),
            nn.Linear(encoder_out_dim, embed_dim),
        )
        self._embed_dim = embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.mlp_head(x)
        x = F.normalize(x, dim=1)
        return x


class RandomTest(nn.Module):
    def __init__(self, dim: int = 128):
        super(RandomTest, self).__init__()
        self.dim = dim

    @property
    def rep_dim(self) -> int:
        return self.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rand((x.shape[0], self.dim), device=x.device)


def make_embeddings(
        model: Embedder,
        dataset: CompoundDataset,
        eval_transforms: Callable,
        plate_sampling_strategy: bool,
        sample_size: int,
        eval_batch_size: int = 128,
) -> np.ndarray:
    df = dataset.get_df()
    embeddings = np.zeros((len(df), model.embed_dim), dtype=np.float32)
    for plate in df['plate'].unique():
        plate_embeddings = []
        plate_indices = df[df['plate'] == plate].index.tolist()
        dataset_filtered_by_plate = IndexFilteredDataset(dataset, plate_indices)

        if plate_sampling_strategy:
            # compute the batch norm layer statistics on controls for this plate
            plate_df = dataset_filtered_by_plate.get_df()
            controls = []
            control_indices = plate_df[plate_df['moa'] == 'DMSO'].index
            for idx in control_indices:
                controls.append(dataset_filtered_by_plate[idx])

            model.train()
            controls, _ = default_collate(controls)
            if controls.shape[0] > sample_size:
                controls = controls[torch.randperm(controls.shape[0])[:sample_size]]
            with torch.no_grad():
                model(eval_transforms(controls))

        # no shuffle, so order of embeddings matches order of plate indices
        dataloader = DataLoader(
            dataset=dataset_filtered_by_plate,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        model.eval()
        with torch.no_grad():
            for (ims, _) in tqdm(dataloader):
                ims = eval_transforms(ims)
                z = model(ims)
                plate_embeddings += z.detach().cpu().numpy().tolist()
        embeddings[plate_indices] = np.array(plate_embeddings)

    return embeddings




