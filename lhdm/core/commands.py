import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import typer
from pytorch_lightning.loggers import WandbLogger
import matplotlib

from core.utils import load_config, get_device
from models.inr import INR
from models.autoencoder import Autoencoder
from data.irn_dataset import DataHandler, DataSelector, DatasetType

matplotlib.use("TkAgg")

cmd = typer.Typer(pretty_exceptions_show_locals=False)


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


@cmd.command()
def visualize_mlp(experiment: str = "autoencoder_sanity_check"):
    """Visualize the MLK model output."""
    cfg = load_config(experiment)
    device = get_device()
    mlp = INR(up_scale=16)

    data_handler = DataHandler(
        hparams={**cfg["data"], "device": device},
        data_folder=cfg["data"]["data_path"],
        selectors=DataSelector(
            dataset_type=DatasetType[cfg["data"]["dataset_type"]],
            class_label=cfg["data"]["class_label"],
            sample_id=cfg["data"]["sample_id"],
        ),
    )

    state_dict = data_handler.get_state_dict(index=0)
    mlp.load_state_dict(state_dict)
    plot_image(mlp)


@cmd.command()
def train(experiment: str = "autoencoder_sanity_check"):
    """Train the autoencoder model."""
    cfg = load_config(experiment)
    device = get_device()
    wandb_logger = WandbLogger(log_model="all")

    data_handler = DataHandler(
        hparams={**cfg["data"], "device": device},
        data_folder=cfg["data"]["data_path"],
        selectors=DataSelector(
            dataset_type=DatasetType[cfg["data"]["dataset_type"]],
            class_label=cfg["data"]["class_label"],
            sample_id=cfg["data"]["sample_id"],
        ),
    )

    train_loader = data_handler.train_dataloader()
    val_loader = data_handler.val_dataloader()

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=cfg["trainer"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg["trainer"]["precision"],
        log_every_n_steps=cfg["trainer"]["log_every_n_steps"],
    )

    model = Autoencoder(
        ddconfig={
            "input_dim": cfg["model"]["input_dim"],
            "output_dim": cfg["model"]["output_dim"],
            "hidden_dim": cfg["model"]["hidden_dim"],
            "z_dim": cfg["model"]["z_dim"],
        },
        embed_dim=cfg["model"]["z_dim"],
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
