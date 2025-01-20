from matplotlib import pyplot as plt
from torch import Tensor
import torch
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


def duplicate_batch_to_size(batch):
    """
    Duplicates elements in a batch until it reaches the target batch size.
    Works with both single tensors and dictionary/tuple batch structures.
    """
    target_batch_size = 2048

    if isinstance(batch, torch.Tensor):
        # For single tensor batch
        current_size = len(batch)
        if current_size >= target_batch_size:
            return batch[:target_batch_size]

        # Calculate how many full copies we need and the remainder
        num_copies = target_batch_size // current_size
        remainder = target_batch_size % current_size

        # Create full copies and add the remainder
        duplicated = batch.repeat(num_copies, *(1 for _ in range(len(batch.shape) - 1)))
        if remainder > 0:
            duplicated = torch.cat([duplicated, batch[:remainder]], dim=0)

        return duplicated

    elif isinstance(batch, dict):
        # For dictionary of tensors
        return {
            k: duplicate_batch_to_size(v, target_batch_size) for k, v in batch.items()
        }

    elif isinstance(batch, (tuple, list)):
        # For tuple/list of tensors
        return type(batch)(duplicate_batch_to_size(x, target_batch_size) for x in batch)

    raise TypeError(f"Unsupported batch type: {type(batch)}")
