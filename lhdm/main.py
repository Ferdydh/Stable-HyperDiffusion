import typer
import pytorch_lightning as pl
import torch

from models import Autoencoder
import sys

sys.path.append('./data')

from irn_dataset import DataHandler

app = typer.Typer()


@app.command()
def test(name: str):
    raise NotImplementedError("Implement this function")


@app.command()
def train():
    # TODO: accept config to run, for now just run the default
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    hparams = {'split_ratio': [80,10,10],
               'device': device,
               'batch_size': 8,
               'num_workers': 4}

    data_handler = DataHandler(hparams, "data/mnist-inrs", "cifar10_png_train_airplane_")
    train_loader = data_handler.train_dataloader()
    val_loader = data_handler.val_dataloader()

    trainer = pl.Trainer()

    ddconfig = {
        "double_z": True,
        "z_channels": 64,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4, 4],  # num_down = len(ch_mult)-1
        "num_res_blocks": 2,
        "attn_resolutions": [16, 8],
        "dropout": 0.0,
    }

    model = Autoencoder(ddconfig=ddconfig, embed_dim=256)
    # model = Identity()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    app()
