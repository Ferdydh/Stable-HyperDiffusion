from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Any

from inr import INR


def load_config(experiment_name: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path("lhdm") / Path("configs") / f"{experiment_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_image(mlp_model: INR) -> None:
    resolution = 28
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    grid_x, grid_y = np.meshgrid(x, y)

    inputs = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

    with torch.no_grad():
        outputs = mlp_model(inputs_tensor).numpy()

    image = outputs.reshape(resolution, resolution)

    plt.imshow(image, cmap="gray", extent=(-1, 1, -1, 1))
    plt.colorbar(label="Grayscale Value")
    plt.title("Generated Grayscale Image")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.ion()
    plt.show(block=True)
