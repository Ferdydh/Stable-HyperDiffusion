import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import typer
from pytorch_lightning.loggers import WandbLogger
import matplotlib

from models.inr import INR
from models.autoencoder import Autoencoder
from data.irn_dataset import DataHandler


matplotlib.use("TkAgg")  # Option 1

wandb_logger = WandbLogger(log_model="all")

cmd = typer.Typer(pretty_exceptions_show_locals=False)


def plot_image(mlp_model):
    # Generate a 28x28 grid of (x, y) inputs
    resolution = 28
    x = np.linspace(-1, 1, resolution)  # Normalize to range [-1, 1]
    y = np.linspace(-1, 1, resolution)
    grid_x, grid_y = np.meshgrid(x, y)

    # Flatten the grid into a list of (x, y) pairs
    inputs = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    # print(inputs)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

    # Pass the inputs through the MLP
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = mlp_model(inputs_tensor).numpy()

    # Reshape the outputs into a 28x28 image
    image = outputs.reshape(resolution, resolution)

    # Plot the image
    plt.imshow(image, cmap="gray", extent=(-1, 1, -1, 1))
    plt.colorbar(label="Grayscale Value")
    plt.title("Generated Grayscale Image")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.ion()  # Turn on interactive mode

    typer.echo(f"Matplotlib backend: {matplotlib.get_backend()}")
    typer.echo(f"Interactive mode: {plt.isinteractive()}")

    plt.show(block=True)
    plt.savefig("temp.png")


@cmd.command()
def visualize_mlp():
    mlp = INR(up_scale=16)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hparams = {
        "split_ratio": [80, 10, 10],
        "device": device,
        "batch_size": 8,
        "num_workers": 4,
        "sample_limit": 1,
        "input_dim": 1185,
        "output_dim": 1185,
        "hidden_dim": 512,
        "z_dim": 64,
    }
    datahandler = DataHandler(hparams, "data/mnist-inrs", "mnist_png_training_9")

    state_dict = datahandler.get_state_dict(index=0)

    mlp.load_state_dict(state_dict)

    plot_image(mlp)


@cmd.command()
def train():
    # TODO: accept config to run, for now just run the default
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hparams = {
        "split_ratio": [80, 10, 10],
        "device": device,
        "batch_size": 8,
        "num_workers": 4,
        "sample_limit": 1,
    }

    data_handler = DataHandler(
        hparams, "data/mnist-inrs", "cifar10_png_train_airplane_"
    )
    train_loader = data_handler.train_dataloader()
    val_loader = data_handler.val_dataloader()

    trainer = pl.Trainer(logger=wandb_logger, max_epochs=5)

    ddconfig = {
        "input_dim": 1185,
        "output_dim": 1185,
        "hidden_dim": 64,
        "z_dim": 16,
    }

    model = Autoencoder(ddconfig=ddconfig, embed_dim=16)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
