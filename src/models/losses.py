import torch.nn.functional as F


def mse_loss(inputs, reconstructions):
    """Mean squared error loss for reconstruction."""
    return F.mse_loss(reconstructions, inputs)
