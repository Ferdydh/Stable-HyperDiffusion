from typing import Dict, Tuple, Any
import torch
import pytorch_lightning as pl
from torch import Tensor
from jaxtyping import Float
from typeguard import typechecked

from models.losses import normal_kl, mse_loss
from .base import Encoder, Decoder


class Autoencoder(pl.LightningModule):
    @typechecked
    def __init__(self, ddconfig: dict, embed_dim: int):
        super().__init__()
        assert ddconfig["double_z"]

        self.encoder: Encoder = Encoder(**ddconfig)
        self.decoder: Decoder = Decoder(**ddconfig)
        self.embed_dim: int = embed_dim
        self.learning_rate: float = 1e-3  # Specify or set this elsewhere as needed

    @classmethod
    @typechecked
    def init_from_ckpt(cls, path: str) -> "Autoencoder":
        sd = torch.load(path, map_location="cpu")["state_dict"]
        print(f"Restored from {path}")
        return cls(
            encoder=Encoder(**sd["encoder"]),
            decoder=Decoder(**sd["decoder"]),
            embed_dim=sd["embed_dim"],
        )

    @typechecked
    def encode(self, x: Float[Tensor, "batch feature_dim"]) -> Any:
        return self.encoder(x)

    @typechecked
    def decode(
        self, z: Float[Tensor, "batch latent_dim"]
    ) -> Float[Tensor, "batch feature_dim"]:
        return self.decoder(z)

    @typechecked
    def forward(
        self, input: Float[Tensor, "batch feature_dim"], sample_posterior: bool = True
    ) -> Tuple[Float[Tensor, "batch feature_dim"], Any]:
        posterior = self.encode(input)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    @typechecked
    def compute_loss(
        self,
        inputs: Float[Tensor, "batch feature_dim"],
        reconstructions: Float[Tensor, "batch feature_dim"],
        posterior: Any,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        mean1, logvar1 = posterior.mean, posterior.logvar
        kl_loss: Tensor = normal_kl(mean1, logvar1)
        recon_loss: Tensor = mse_loss(inputs, reconstructions)

        # Return combined VAE loss (KL + reconstruction)
        total_loss: Tensor = kl_loss + recon_loss
        return total_loss, {"train/kl_loss": kl_loss, "train/rec_loss": recon_loss}

    @typechecked
    def training_step(
        self, batch: Float[Tensor, "batch feature_dim"], batch_idx: int
    ) -> Tensor:
        inputs = batch.float()  # Process the batch directly
        reconstructions, posterior = self(inputs)
        loss, log_dict = self.compute_loss(inputs, reconstructions, posterior)

        self.log(
            "aeloss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        self.log_dict(
            log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False
        )
        return loss

    @typechecked
    def validation_step(
        self, batch: Float[Tensor, "batch feature_dim"], batch_idx: int
    ) -> Dict[str, Tensor]:
        inputs = batch.float()  # Process the batch directly
        reconstructions, posterior = self(inputs)
        loss, log_dict = self.compute_loss(inputs, reconstructions, posterior)

        self.log("val/rec_loss", log_dict["train/rec_loss"])
        self.log_dict(log_dict)
        return log_dict

    @typechecked
    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate,
            betas=(0.5, 0.9),
        )
        return opt_ae

    @typechecked
    def get_last_layer(self) -> Float[Tensor, "..."]:
        return self.decoder.conv_out.weight
