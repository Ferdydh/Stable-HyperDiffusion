import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from core.utils import load_config, get_device, plot_image
from models.inr import INR
from models.autoencoder import Autoencoder
from data.irn_dataset import DataHandler, DataSelector, DatasetType
from core import cmd


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
        accelerator="auto",
        devices="auto",
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
