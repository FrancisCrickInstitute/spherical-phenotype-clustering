import os
from typing import Callable, Dict, Tuple

import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
from umap import UMAP

from model import Embedder
from dataset import CompoundDataset
from visualization import save_interactive_scatter_plot
from dfconst import META_DF_COLUMNS, PLATE_COLUMN, WELL_COLUMN, TREATMENT_COLUMN, MOA_COLUMN, COMPOUND_NAME_COLUMN


def knn1(embeddings: np.ndarray, df: pd.DataFrame, category: str) -> float:
    # make sure embeddings are unit vectors
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    nn = 0
    for idx in range(embeddings.shape[0]):
        dists = np.linalg.norm(embeddings[idx:idx + 1, ...] - embeddings, axis=1)
        indices = np.argpartition(dists, 2)[:2]
        result = df.iloc[indices]
        # one index is the point itself, the other is the closest neighbor
        nn += int(result[category].iloc[0] == result[category].iloc[1])

    return nn / embeddings.shape[0]


def not_same_compound_score(
        embeddings: np.ndarray,
        compound_labels: np.ndarray,
        moa_labels: np.ndarray,
        metric: str = 'cosine',
) -> float:
    same_moa = 0
    total = 0
    for e in range(embeddings.shape[0]):
        emb = embeddings[e, :]

        c = compound_labels[e]
        indices = (compound_labels != c).nonzero()[0]
        nsc = embeddings[indices]

        cdist = sp.spatial.distance.cdist([emb], nsc, metric=metric)
        nn_idx = indices[np.argmin(cdist)]

        moa1 = moa_labels[e]
        moa2 = moa_labels[nn_idx]
        same_moa += int(moa1 == moa2)
        total += 1
    if total == 0:
        return 0
    return same_moa/total


def not_same_compound_batch_score(
        embeddings: np.ndarray,
        compound_labels: np.ndarray,
        moa_labels: np.ndarray,
        batch_labels: np.ndarray,
        metric: str = 'cosine',
) -> float:
    same_moa = 0
    total = 0
    for e in range(embeddings.shape[0]):
        emb = embeddings[e, :]

        moa1 = moa_labels[e]
        c = compound_labels[e]
        b = batch_labels[e]
        indices = ((compound_labels != c) & (batch_labels != b)).nonzero()[0]
        nscb = embeddings[indices]

        cdist = sp.spatial.distance.cdist([emb], nscb, metric=metric)
        nn_idx = indices[np.argmin(cdist)]

        moa2 = moa_labels[nn_idx]
        same_moa += int(moa1 == moa2)
        total += 1
    if total == 0:
        return 0
    return same_moa/total


def make_well_averages(embeddings: np.array, meta_df: pd.DataFrame) -> Tuple[np.array, pd.DataFrame]:
    # well level embeddings
    well_embeddings = []

    print('Generating well level embeddings...')
    well_meta_df = pd.DataFrame(columns=META_DF_COLUMNS)
    for i, (_, indices) in tqdm(enumerate(meta_df.groupby([PLATE_COLUMN, WELL_COLUMN]).indices.items())):
        well_embedding = np.mean(embeddings[indices, :], axis=0)
        # indices[0] since these columns are the same across the well
        meta = meta_df.iloc[indices[0]]
        well_meta_df = pd.concat([well_meta_df, pd.DataFrame.from_records([dict(meta[META_DF_COLUMNS])])])
        well_embeddings.append(well_embedding)

    well_embeddings = np.array(well_embeddings)
    # make sure well embeddings are unit vectors
    well_embeddings = well_embeddings / np.linalg.norm(well_embeddings, axis=1, keepdims=True)
    well_meta_df = well_meta_df.reset_index(drop=True)

    return well_embeddings, well_meta_df


def calculate_bbbc021_metrics(well_embeddings, well_meta_df) -> Tuple:
    """ Treatment level nsc/nscb (BBBC021 only). """
    # single treatment embeddings
    treatment_embeddings = []
    treatment_meta_df = pd.DataFrame(columns=META_DF_COLUMNS)
    for i, (_, indices) in tqdm(enumerate(well_meta_df.groupby([TREATMENT_COLUMN]).indices.items())):
        treatment_embedding = np.median(well_embeddings[indices, :], axis=0)
        treatment_embeddings.append(treatment_embedding)

        meta = well_meta_df.iloc[indices[0]]
        treatment_meta_df = pd.concat([treatment_meta_df, pd.DataFrame.from_records([dict(meta[META_DF_COLUMNS])])])
    treatment_embeddings = np.array(treatment_embeddings)

    # BBBC021 - DMSO and unknown are not used in metric reporting
    nsc_valid_samples = (treatment_meta_df[MOA_COLUMN] != 'unknown') & (treatment_meta_df[MOA_COLUMN] != 'DMSO')
    treatment_embeddings = treatment_embeddings[nsc_valid_samples]
    treatment_meta_df = treatment_meta_df[nsc_valid_samples]

    nsc = not_same_compound_score(
        embeddings=treatment_embeddings,
        compound_labels=treatment_meta_df[COMPOUND_NAME_COLUMN].values,
        moa_labels=treatment_meta_df[MOA_COLUMN].values,
        metric='cosine',
    )

    # Kinase inhibitors and Cholesterol-lowering are only in one batch
    nscb_valid_samples = (treatment_meta_df[MOA_COLUMN] != 'Kinase inhibitors') & (treatment_meta_df[MOA_COLUMN] != 'Cholesterol-lowering')
    nscb_treatment_embeddings = treatment_embeddings[nscb_valid_samples]
    nscb_treatment_meta_df = treatment_meta_df[nscb_valid_samples]

    nscb = not_same_compound_batch_score(
        embeddings=nscb_treatment_embeddings,
        compound_labels=nscb_treatment_meta_df[COMPOUND_NAME_COLUMN].values,
        moa_labels=nscb_treatment_meta_df[MOA_COLUMN].values,
        # week
        batch_labels=nscb_treatment_meta_df[PLATE_COLUMN].astype(str).str.split('_').str[0].values,
        metric='cosine',
    )

    return nsc, nscb


def evaluate_model(
        experiment_folder: str,
        model: Embedder,
        dataset: CompoundDataset,
        embedding_fn: Callable,
        save_visualizations: bool,
        save_embeddings: bool,
        include_bbbc021_metrics: bool = False,
        prefix: str = '',
) -> Dict[str, float]:
    embeddings = embedding_fn(model, dataset)
    well_embeddings, well_meta_df = make_well_averages(embeddings, dataset.get_df())

    metrics = {}
    # 1-NN to moa
    metrics[f'{prefix}well-1nn-moa'] = knn1(well_embeddings, well_meta_df, MOA_COLUMN)
    # 1-NN to treatment
    metrics[f'{prefix}well-1nn-treatment'] = knn1(well_embeddings, well_meta_df, TREATMENT_COLUMN)

    if include_bbbc021_metrics:
        nsc, nscb = calculate_bbbc021_metrics(well_embeddings, well_meta_df)
        metrics[f'{prefix}nsc'] = nsc
        metrics[f'{prefix}nscb'] = nscb

    if save_visualizations:
        tsne = TSNE(n_components=2, metric='cosine', perplexity=30.0)
        tsne12 = tsne.fit_transform(well_embeddings)
        um = UMAP(n_components=2, metric='cosine', min_dist=0.5, n_neighbors=20)
        um12 = um.fit_transform(well_embeddings)

        save_interactive_scatter_plot(
            save_fpath=os.path.join(experiment_folder, f'{prefix}tsne.html'),
            embeddings2d=tsne12,
            meta_df=well_meta_df,
            plot_label_type=MOA_COLUMN,
            hover_label_types=META_DF_COLUMNS,
        )

        save_interactive_scatter_plot(
            save_fpath=os.path.join(experiment_folder, f'{prefix}umap.html'),
            embeddings2d=um12,
            meta_df=well_meta_df,
            plot_label_type=MOA_COLUMN,
            hover_label_types=META_DF_COLUMNS,
        )

    if save_embeddings:
        np.save(os.path.join(experiment_folder, f'{prefix}embeddings.npy'), embeddings)

    return metrics



