from matplotlib import pyplot as plt
import numpy as np
import torch

from src.data.inr import INR


def plot_image(
    mlp_model: INR, device: torch.device
):# -> plt.Figure:  # Updated return type hint
    resolution = 28
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    grid_x, grid_y = np.meshgrid(x, y)

    inputs = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = mlp_model(inputs_tensor).cpu().numpy()

    image = outputs.reshape(resolution, resolution)

    #fig, ax = plt.subplots()
    #ax.imshow(image, cmap="gray", extent=(-1, 1, -1, 1))
    #plt.axis("off")
    #return fig  # Returns the matplotlib Figure object
    return image
