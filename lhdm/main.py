import typer
import pytorch_lightning as pl

from models import Autoencoder

app = typer.Typer()


@app.command()
def test(name: str):
    raise NotImplementedError("Implement this function")


@app.command()
def train():
    # TODO: accept config to run, for now just run the default
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
    trainer.fit(model=model, train_dataloaders=None, val_dataloaders=None)


if __name__ == "__main__":
    app()
