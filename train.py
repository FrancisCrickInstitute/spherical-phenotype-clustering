import os
import time
from typing import Callable, Dict, List, Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import NTXent, get_updated_hypersphere_position
from dataset import CompoundDataset
from evaluate import evaluate_model
from model import Embedder


def train_one_epoch(
        model: torch.nn.Module,
        dataset: CompoundDataset,
        epoch: int,
        max_epoch: int,
        batch_size: int,
        loss_func: NTXent,
        optimizer: torch.optim.Optimizer,
        train_transforms: Callable,
        wells: torch.Tensor,
        dataset_idx_to_well_idx: torch.Tensor,
) -> float:
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    running_loss = 0
    with tqdm(total=len(data_loader), desc=f'Epoch: {epoch}/{max_epoch}') as progress:
        for step, (x, idx) in enumerate(data_loader):
            # for each sample in the batch this gives the well index
            batch_idx_to_well_idx = dataset_idx_to_well_idx[idx]
            # apply augmentations sample by sample
            x = torch.stack([train_transforms(x_i) for x_i in x], dim=0)

            optimizer.zero_grad()

            y_instance = model(x)
            y_well = wells[batch_idx_to_well_idx]

            loss = loss_func(y_instance, y_well)

            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

            # move wells along surface
            with torch.no_grad():
                for b_idx, well_idx in enumerate(batch_idx_to_well_idx):
                    wells[well_idx] = get_updated_hypersphere_position(wells[well_idx], y_instance[b_idx, ...])

            progress.set_postfix({'loss': f"{running_loss / (step + 1):.3f}"})
            progress.update()
    return running_loss / len(data_loader)


def log_epoch(experiment_folder: str, epoch_data: Dict):
    log_fpath = os.path.join(experiment_folder, 'log.csv')
    if not os.path.exists(log_fpath):
        columns = list(epoch_data.keys())
        with open(log_fpath, 'w') as f:
            f.write(','.join(columns) + '\n')
    values = [str(v) for v in epoch_data.values()]
    with open(log_fpath, 'a') as f:
        f.write(','.join(values) + '\n')


def save_model(
        experiment_folder: str,
        model: torch.nn.Module,
        epoch: int,
        performance: float,
        metric: str = 'loss'
):
    # remove previous best model for this metric
    model_files = os.listdir(experiment_folder)
    for file in model_files:
        filename, ext = os.path.splitext(file)
        if ext == '.pt':
            old_metric, _ = filename.split('.', 1)
            if old_metric == metric:
                os.remove(os.path.join(experiment_folder, file))
    # save new best model
    encoded_performance = f"{performance:.4f}".replace('.', 'P')
    save_filename = f"{metric}.{epoch}.{encoded_performance}.pt"
    save_fpath = os.path.join(experiment_folder, save_filename)
    torch.save(model.state_dict(), save_fpath)


def load_model(model: Embedder, experiment_folder: str, early_stopping_metric: str) -> Embedder:
    for file in os.listdir(experiment_folder):
        filename, ext = os.path.splitext(file)

        if ext == '.pt':
            parts = filename.split('.', 2)
            old_metric, old_epoch, old_performance = parts
            old_performance = float(old_performance.replace('P', '.'))
            old_epoch = int(old_epoch)

            if old_metric == early_stopping_metric:
                print(f"Found previous model at epoch={old_epoch}, {old_metric}={old_performance:.4f}, reloading.")
                model.load_state_dict(torch.load(os.path.join(experiment_folder, file)))
                return model

    print(f"No previous models found.")
    return model


def train_model(
        experiment_folder: str,
        model: Embedder,
        embedding_fn: Callable,
        dataset: CompoundDataset,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        batch_size: int,
        temperature: float,
        train_transforms: Callable,
        device: torch.device,
        embed_dim: int,
        early_stopping_metric: str = 'loss',
        include_bbbc021_metrics: bool = False,
        init_wells_with_model: bool = False,
        group_images_by: Union[List, Tuple] = ('plate', 'well'),
) -> Embedder:
    best_loss = float('inf')
    best_metric_performance = -float('inf')

    df = dataset.get_df()
    well_groups = df.groupby(group_images_by).groups
    dataset_idx_to_well_idx = torch.zeros(len(dataset), dtype=torch.long)
    for g, (group, indices) in enumerate(well_groups.items()):
        dataset_idx_to_well_idx[indices] = g

    # initialize wells as random points on the unit sphere
    wells = torch.rand((len(well_groups), embed_dim), dtype=torch.float32).to(device)
    if init_wells_with_model:
        # if requested, initialize wells with the model's embeddings
        embeds = embedding_fn(model, dataset)
        for g, (_, indices) in enumerate(well_groups.items()):
            wells[g] = torch.tensor(embeds[indices]).mean(dim=0).to(device)
    wells = wells / wells.norm(dim=1, keepdim=True)

    loss_func = NTXent(temperature=temperature)

    loss = 0.0
    for epoch in range(1, n_epochs+1):
        start_epoch_time = time.time()

        model.train()
        loss = train_one_epoch(
            model=model,
            dataset=dataset,
            epoch=epoch,
            max_epoch=n_epochs,
            batch_size=batch_size,
            loss_func=loss_func,
            optimizer=optimizer,
            train_transforms=train_transforms,
            wells=wells,
            dataset_idx_to_well_idx=dataset_idx_to_well_idx,
        )
        end_epoch_time = time.time()

        model.eval()
        metrics = evaluate_model(
            experiment_folder=experiment_folder,
            model=model,
            dataset=dataset,
            embedding_fn=embedding_fn,
            save_visualizations=False,
            save_embeddings=False,
            include_bbbc021_metrics=include_bbbc021_metrics,
        )

        print(f"Metrics: {metrics}")
        print()
        end_metrics_time = time.time()

        current_time = time.strftime("%H:%M:%S", time.gmtime())
        log_epoch(experiment_folder, {
            'epoch': epoch,
            'time': current_time,
            'train_time': end_epoch_time - start_epoch_time,
            'eval_time': end_metrics_time - end_epoch_time,
            'loss': loss,
            **metrics,
        })

        for metric, performance in metrics.items():
            if metric == early_stopping_metric and performance > best_metric_performance:
                save_model(experiment_folder, model, epoch, performance, metric)
                best_metric_performance = performance
        if loss < best_loss:
            save_model(experiment_folder, model, epoch, loss, 'loss')
            best_loss = loss

    # early stopping
    if early_stopping_metric is not None:
        model = load_model(model, experiment_folder, early_stopping_metric)
    save_model(experiment_folder, model, n_epochs, loss, 'end')

    return model


