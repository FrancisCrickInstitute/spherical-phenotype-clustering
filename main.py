import os
import sys
import functools
from typing import Optional

import torch
import yaml

from dataset import CompoundDataset, CompoundTensorDataset
from model import Embedder, make_embeddings, RandomTest
from resnet import resnet18
from train import train_model
from evaluate import evaluate_model
from transforms import load_transforms


EXPERIMENTS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'experiments'))


def main(config_filepath: Optional[str] = None):
    if config_filepath is None and len(sys.argv) == 1:
        raise ValueError("Must provide config filepath as command line argument or as first argument to main()")

    if config_filepath is None:
        config_filepath = sys.argv[1]

    with open(config_filepath) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    experiment_name, _ = os.path.splitext(os.path.basename(config_filepath))
    experiment_folder = os.path.join(EXPERIMENTS_FOLDER, experiment_name)

    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    # copy config file to experiment folder
    config_filename = os.path.basename(config_filepath)
    config_dest = os.path.join(experiment_folder, config_filename)
    with open(config_dest, 'w') as f:
        yaml.dump(config, f)


    tensor = torch.load(config['dataset']['tensor_filepath'])

    dataset = CompoundTensorDataset(
        csv_fpath=config['dataset']['csv_filepath'],
        tensor=tensor,
    )

    eval_dataset = None
    if 'eval_dataset' in config:
        eval_dataset = CompoundDataset(
            csv_fpath=config['eval_dataset']['csv_filepath'],
            max_cache_size=None,
        )

    # check an image can be loaded
    test_im, _ = dataset[0]
    n_channels = test_im.shape[0]

    norm_layer = torch.nn.BatchNorm2d
    if 'batch_norm' in config['model']:
        norm_layer = functools.partial(
            torch.nn.BatchNorm2d,
            track_running_stats=config['model']['batch_norm']['track_running_stats'],
            momentum=config['model']['batch_norm']['momentum'],
            affine=config['model']['batch_norm']['affine'],
        )

    embed_dim = config['model']['embed_dim']
    encoder_type = config['model']['encoder']
    if encoder_type == 'ResNet18':
        encoder = resnet18(norm_layer=norm_layer, in_channels=n_channels)
    elif encoder_type == 'RandomTest':
        encoder = RandomTest()
    else:
        raise ValueError(f"Unknown encoder: {encoder_type}")
    model = Embedder(
        encoder=encoder,
        encoder_out_dim=encoder.rep_dim,
        embed_dim=embed_dim,
    )

    # now set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer_type = config['train_conf']['optimizer']['type']
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=config['train_conf']['optimizer']['lr'],
            weight_decay=config['train_conf']['optimizer']['wd'],
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    train_transforms = load_transforms(config['train_conf']['transforms'], device)
    eval_transforms = load_transforms(config['eval_conf']['transforms'], device)

    plate_sampling_strategy = bool(config['eval_conf']['sampling_strategy']['type'] == 'plate')
    embedding_fn = functools.partial(
        make_embeddings,
        eval_transforms=eval_transforms,
        plate_sampling_strategy=plate_sampling_strategy,
        sample_size=config['eval_conf']['sampling_strategy']['sample_size'],
        eval_batch_size=128,
    )

    include_bbbc021_metrics = config['eval_conf'].get('include_bbbc021_metrics', False)

    init_wells_with_model = config['train_conf'].get('init_wells_with_model', False)
    early_stopping_metric = config['train_conf'].get('early_stopping_metric', None)
    group_images_by = config['train_conf'].get('group_images_by', ['plate', 'well'])

    model = train_model(
        experiment_folder=experiment_folder,
        model=model,
        embedding_fn=embedding_fn,
        dataset=dataset,
        optimizer=optimizer,
        n_epochs=config['train_conf']['n_epochs'],
        batch_size=config['train_conf']['batch_size'],
        temperature=config['train_conf']['temperature'],
        train_transforms=train_transforms,
        device=device,
        embed_dim=embed_dim,
        early_stopping_metric=early_stopping_metric,
        include_bbbc021_metrics=include_bbbc021_metrics,
        init_wells_with_model=init_wells_with_model,
        group_images_by=group_images_by,
    )

    # eval
    metrics = evaluate_model(
        experiment_folder=experiment_folder,
        model=model,
        dataset=dataset,
        embedding_fn=embedding_fn,
        save_visualizations=True,
        save_embeddings=config['eval_conf']['save_embeddings'],
        include_bbbc021_metrics=include_bbbc021_metrics,
        prefix='train_',
    )

    if eval_dataset is not None:
        eval_metrics = evaluate_model(
            experiment_folder=experiment_folder,
            model=model,
            dataset=eval_dataset,
            embedding_fn=embedding_fn,
            save_visualizations=True,
            save_embeddings=config['eval_conf']['save_embeddings'],
            include_bbbc021_metrics=include_bbbc021_metrics,
            prefix='eval_',
        )
        metrics.update(eval_metrics)

    print("Final metrics:")
    print(metrics)

    # save metrics
    metrics_fpath = os.path.join(experiment_folder, 'metrics.yaml')
    with open(metrics_fpath, 'w') as f:
        yaml.dump(metrics, f)


if __name__ == '__main__':
    main()


