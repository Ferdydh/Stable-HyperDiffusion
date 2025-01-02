from matplotlib import pyplot as plt
from torch import Tensor
import wandb

from src.core.visualize import plot_image
from src.data.inr import INR
from src.data.data_converter import flattened_weights_to_weights, tokens_to_weights


def flattened_weights_to_image_dict(
    weights: Tensor, inr_model: INR, prefix, device
) -> dict:
    weights = [flattened_weights_to_weights(w, inr_model) for w in weights]

    return weights_to_image_dict(weights, inr_model, prefix, device)


def tokens_to_image_dict(
    tokens: list, pos, inr_model: INR, prefix, device, reference_checkpoint
):
    weights = [
        tokens_to_weights(t, p, reference_checkpoint) for t, p in zip(tokens, pos)
    ]
    return weights_to_image_dict(weights, inr_model, prefix, device)


def weights_to_image_dict(weights: list, inr_model: INR, prefix, device):
    result_dict = {}

    # Create visualizations for each pair
    for i, recon in enumerate(weights):
        inr_model.load_state_dict(recon)
        recon_fig = plot_image(inr_model, device)

        result_dict[f"{prefix}/{i}"] = wandb.Image(recon_fig)

        plt.close(recon_fig)

    return result_dict
