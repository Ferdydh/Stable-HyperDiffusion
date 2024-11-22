import typer
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from models import Autoencoder
from data.irn_dataset import DataHandler

app = typer.Typer(pretty_exceptions_show_locals=False)


wandb_logger = WandbLogger(log_model="all")


@app.command()
def test(name: str):
    raise NotImplementedError("Implement this function")


@app.command()
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


if __name__ == "__main__":
    app()
