from matplotlib import pyplot as plt
from torch import Tensor
import wandb

from src.core.config import get_device
from src.core.utils import plot_image
from src.data.inr import INR
from src.data.utils import flattened_weights_to_weights


def log_reconstructed_image(
    reconstructions: Tensor,
    inr_model: INR,
    prefix: str,
) -> dict:
    """Log reconstructed image from flattened to wandb."""
    result_dict = {}

    # Create visualizations for each pair
    for i, recon in enumerate(reconstructions):
        # Generate figures
        weights = flattened_weights_to_weights(recon, inr_model)
        inr_model.load_state_dict(weights)
        recon_fig = plot_image(inr_model, recon.device)

        result_dict[f"{prefix}/reconstruction_{i}"] = wandb.Image(recon_fig)

        plt.close(recon_fig)

    return result_dict


def log_original_image(
    originals: Tensor,
    inr_model: INR,
    prefix: str,
) -> dict:
    """Log original image to wandb."""
    result_dict = {}

    for i, orig in enumerate(originals):
        # Generate figure
        weights = flattened_weights_to_weights(orig, inr_model)
        inr_model.load_state_dict(weights)
        original_fig = plot_image(inr_model, orig.device)

        # Add to result dictionary with unique keys
        result_dict[f"{prefix}/original_{i}"] = wandb.Image(original_fig)

    # Close figure
    plt.close(original_fig)

    return result_dict
