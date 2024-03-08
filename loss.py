from math import pi

import torch
from torch import nn
from torch import arcsin, tan, sqrt


DEFAULT_UPDATE_ANGLE = pi / 160.


def get_updated_hypersphere_position(
        p: torch.Tensor,
        q: torch.Tensor,
        dphi: float = DEFAULT_UPDATE_ANGLE,
) -> torch.Tensor:
    """ Move p along a geodesic of the hypersphere towards q. """
    d = torch.linalg.vector_norm(q - p)
    phi = max(-arcsin(d/2.), arcsin(d/2.) - dphi)
    p += (d/2. - sqrt(1 - d**2/4.)*tan(phi))*(q - p)/d
    return p / torch.linalg.vector_norm(p)


class NTXent(nn.Module):
    """ Adapted from: https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/losses/losses.py """
    def __init__(self, temperature):
        super(NTXent, self).__init__()
        self.temperature = temperature

    def forward(self, features, contrast_features):
        mask = torch.eye(features.shape[0], dtype=torch.float32).to(features.device)
        dot_product = torch.matmul(features, torch.cat((features, contrast_features), dim=0).T) / self.temperature

        # log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0]).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss


