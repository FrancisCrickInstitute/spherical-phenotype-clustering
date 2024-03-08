from typing import List, Dict, Callable

import torch
from torchvision import transforms as vision_transforms


def im255center(im: torch.Tensor) -> torch.Tensor:
    return (im - 127.5) / 255.


def load_transforms(transforms_config: List[Dict], device: torch.device) -> Callable:
    transforms = []
    for t_config in transforms_config:
        transform_type = t_config['type']
        if transform_type == 'RandomResizedCrop':
            size = t_config.get('size', 28)
            scale = t_config.get('scale', (0.5, 1.0))
            transform = vision_transforms.RandomResizedCrop(size=size, scale=scale)
        elif transform_type == 'RandomHorizontalFlip':
            p = t_config.get('p', 0.5)
            transform = vision_transforms.RandomHorizontalFlip(p=p)
        elif transform_type == 'RandAugment':
            num_ops = t_config.get('num_ops', 2)
            magnitude = t_config.get('magnitude', 9)
            transform = vision_transforms.RandAugment(num_ops=num_ops, magnitude=magnitude)
        elif transform_type == 'GaussianBlur':
            p = t_config.get('p', 0.5)
            kernel_size = t_config['kernel_size']
            sigma = t_config['sigma']
            transform = vision_transforms.RandomApply([
                vision_transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
            ], p=p)
        elif transform_type == 'ToDevice':
            transform = lambda x: x.to(device)
        elif transform_type == 'im255center':
            transform = im255center
        elif transform_type == 'ToFloat32':
            transform = lambda x: x.to(torch.float32)
        else:
            raise ValueError(f"Unrecognised transform type: {transform_type}")
        transforms.append(transform)
    return vision_transforms.Compose(transforms)


