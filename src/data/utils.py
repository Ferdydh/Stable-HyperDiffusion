from typing import List
import os

import numpy as np
import torch

from src.models.utils import flattened_weights_to_weights

from src.core.config import (
    DataSelector,
    DatasetType,
)


def get_files_from_selectors(data_path, selectors: List[DataSelector]) -> List[str]:
    """Get all files matching the given selectors."""
    all_files = []

    for selector in selectors:
        matching_files = []

        # List all files in the data directory
        for f in os.listdir(data_path):
            # If no dataset type specified, match both MNIST and CIFAR10
            if selector.dataset_type is None:
                if not (f.startswith("mnist_png_") or f.startswith("cifar10_png_")):
                    continue
            # For MNIST: match mnist_png_(train|test)_digit_id pattern
            elif selector.dataset_type == DatasetType.MNIST:
                if not f.startswith("mnist_png_"):
                    continue
            # For CIFAR10: match cifar10_png_train_class_id pattern
            elif selector.dataset_type == DatasetType.CIFAR10:
                if not f.startswith("cifar10_png_train"):
                    continue

            # If class label is specified, match only that class/digit
            if selector.class_label is not None:
                if f"_{selector.class_label}_" not in f:
                    continue

            # If sample_id is specified, match that specific ID
            if selector.sample_id is not None:
                if not f.endswith(f"_{selector.sample_id}"):
                    continue

            # Add the full path to the model file
            model_path = os.path.join(
                os.path.join(os.path.join(data_path, f), "checkpoints"),
                "model_final.pth",
            )
            if os.path.exists(model_path):
                matching_files.append(model_path)

        all_files.extend(matching_files)

    return sorted(list(set(all_files)))


def generate_images(samples, mlp, device):
        image_samples = []
        for weights in samples:
            state_dict = flattened_weights_to_weights(weights, mlp)
            mlp.load_state_dict(state_dict)
            resolution = 28
            x = np.linspace(-1, 1, resolution)
            y = np.linspace(-1, 1, resolution)
            grid_x, grid_y = np.meshgrid(x, y)

            inputs = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)

            with torch.no_grad():
                outputs = mlp(inputs_tensor)
            
            image = outputs.reshape(resolution, resolution).to(torch.uint8)
            image_samples.append(image)
        return torch.stack(image_samples)