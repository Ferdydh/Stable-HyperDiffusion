from matplotlib import pyplot as plt
from torch import Tensor
import torch.nn as nn
from typeguard import typechecked
import wandb
from core.utils import plot_image
from models.inr import INR


def load_weights_into_inr(weights: Tensor, inr_model: INR) -> INR:
    """Helper function to load weights into INR model."""
    state_dict = {}
    start_idx = 0
    for key, param in inr_model.state_dict().items():
        param_size = param.numel()
        param_data = weights[start_idx : start_idx + param_size].reshape(param.shape)
        state_dict[key] = param_data
        start_idx += param_size
    inr_model.load_state_dict(state_dict)
    return inr_model


def create_reconstruction_visualizations(
    originals: Tensor,
    reconstructions: Tensor,
    inr_model: INR,
    prefix: str,
    batch_idx: int,
    global_step: int,
    is_fixed: bool = False,
) -> dict:
    """Create visualization grid for original-reconstruction pairs."""
    result_dict = {}

    # Create visualizations for each pair
    for i, (orig, recon) in enumerate(zip(originals, reconstructions)):
        # Generate figures
        original_fig = plot_image(load_weights_into_inr(orig, inr_model), orig.device)
        recon_fig = plot_image(load_weights_into_inr(recon, inr_model), recon.device)

        # Add to result dictionary with unique keys
        sample_type = "fixed" if is_fixed else "batch"
        result_dict[f"{prefix}/{sample_type}/original_{i}"] = wandb.Image(original_fig)
        result_dict[f"{prefix}/{sample_type}/reconstruction_{i}"] = wandb.Image(
            recon_fig
        )

        # Close figures
        plt.close(original_fig)
        plt.close(recon_fig)

    return result_dict


@typechecked
def get_activation(activation_name: str) -> nn.Module:
    """Helper function to map activation names to PyTorch functions."""
    activations = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    return activations.get(activation_name.lower(), nn.ReLU)()
