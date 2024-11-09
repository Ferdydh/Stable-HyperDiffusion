import torch
import torch.nn.functional as F


def normal_kl(mean1, logvar1, mean2=0.0, logvar2=0.0):
    """Compute the KL divergence between two Gaussians."""
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(mean1.device)
        for x in (logvar1, logvar2)
    ]
    return (
        0.5
        * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        ).mean()
    )


def mse_loss(inputs, reconstructions):
    """Mean squared error loss for reconstruction."""
    return F.mse_loss(reconstructions, inputs)
