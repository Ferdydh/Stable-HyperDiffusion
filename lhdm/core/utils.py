from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Any

from models.inr import INR


def load_config(experiment_name: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path("lhdm") / Path("configs") / f"{experiment_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_image(
    mlp_model: INR, device: torch.device
) -> plt.Figure:  # Updated return type hint
    resolution = 28
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    grid_x, grid_y = np.meshgrid(x, y)

    inputs = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = mlp_model(inputs_tensor).cpu().numpy()

    image = outputs.reshape(resolution, resolution)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray", extent=(-1, 1, -1, 1))
    plt.axis("off")
    return fig  # Returns the matplotlib Figure object
