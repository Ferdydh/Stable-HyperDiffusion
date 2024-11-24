from matplotlib import pyplot as plt
import torch
import pytorch_lightning as pl
import wandb
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
from jaxtyping import Float
from typeguard import typechecked

from models.base import Decoder, Encoder
from models.inr import INR
from core.utils import plot_image


class Autoencoder(pl.LightningModule):
    @typechecked
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Initialize encoder and decoder
        self.encoder = Encoder(**config["model"])
        self.decoder = Decoder(**config["model"])

        # Initialize fixed validation samples
        self.fixed_samples: list[Tensor] | None = None
        self.fixed_sample_reconstructions: dict[str, list[Tensor]] = {}

        # Store optimizer and scheduler config
        self.optimizer_config = config["optimizer"]
        self.scheduler_config = config["scheduler"]

        # Initialize quality metrics
        self.best_val_loss = float("inf")

        # Create demo INR for visualization
        self.demo_inr = INR(up_scale=16)
        # Move demo INR to the same device as the model
        self.demo_inr = self.demo_inr.to(self.device)

    def setup(self, stage: str | None = None):
        """Setup fixed validation samples for tracking reconstruction progress."""
        if stage == "fit" and self.fixed_samples is None:
            try:
                if (
                    hasattr(self.trainer, "val_dataloaders")
                    and self.trainer.val_dataloaders is not None
                ):
                    # Get samples from first validation batch
                    val_batch = next(iter(self.trainer.val_dataloaders[0]))
                    num_samples = min(
                        self.config["logging"]["num_samples_to_visualize"],
                        val_batch.size(0),
                    )
                    self.fixed_samples = val_batch[:num_samples].clone()
            except Exception as e:
                print(f"Warning: Could not setup fixed validation samples: {e}")
                self.fixed_samples = None

    @typechecked
    def encode(self, x: Float[Tensor, "batch feature_dim"]) -> Tensor:
        return self.encoder(x)

    @typechecked
    def decode(
        self, z: Float[Tensor, "batch latent_dim"]
    ) -> Float[Tensor, "batch feature_dim"]:
        return self.decoder(z)

    @typechecked
    def forward(
        self, input: Float[Tensor, "batch feature_dim"]
    ) -> Float[Tensor, "batch feature_dim"]:
        z = self.encode(input)
        dec = self.decode(z)
        return dec

    def compute_loss(
        self,
        inputs: Float[Tensor, "batch feature_dim"],
        reconstructions: Float[Tensor, "batch feature_dim"],
        prefix: str = "train",
    ) -> Tuple[Tensor, dict[str, Tensor]]:
        recon_loss = F.mse_loss(reconstructions, inputs)
        return recon_loss, {f"{prefix}/loss": recon_loss}

    def reconstruct_and_visualize(
        self, weights: Tensor, name: str
    ) -> dict[str, wandb.Image]:
        """Reconstruct image from weights and create visualization."""
        device = weights.device

        # Load weights into demo INR
        state_dict = {}
        start_idx = 0
        for key, param in self.demo_inr.state_dict().items():
            param_size = param.numel()
            param_data = weights[start_idx : start_idx + param_size].reshape(
                param.shape
            )
            state_dict[key] = param_data
            start_idx += param_size

        self.demo_inr.load_state_dict(state_dict)
        plot_image(self.demo_inr, device)

        # Log to wandb and close figure
        wandb_image = wandb.Image(plt)
        plt.close()

        return {f"{name}_reconstruction": wandb_image}

    @typechecked
    def training_step(
        self, batch: Float[Tensor, "batch feature_dim"], batch_idx: int
    ) -> Tensor:
        # Forward pass
        reconstructions = self(batch)
        loss, log_dict = self.compute_loss(batch, reconstructions, prefix="train")

        # Logging
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        # Log gradient norm
        if batch_idx % self.config["trainer"]["log_every_n_steps"] == 0:
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            self.log("train/grad_norm", total_norm, prog_bar=False, sync_dist=True)

        return loss

    @typechecked
    def validation_step(
        self, batch: Float[Tensor, "batch feature_dim"], batch_idx: int
    ) -> dict[str, Tensor]:
        reconstructions = self(batch)
        val_loss, val_log_dict = self.compute_loss(batch, reconstructions, prefix="val")

        # Log validation metrics
        self.log_dict(val_log_dict, prog_bar=True, sync_dist=True)

        # Log current batch samples if no fixed samples are available
        samples_to_log = (
            self.fixed_samples
            if self.fixed_samples is not None
            else batch[: self.config["logging"]["num_samples_to_visualize"]]
        )

        # Log reconstructions periodically
        if batch_idx == 0:
            if (
                self.current_epoch % self.config["logging"]["sample_every_n_epochs"]
                == 0
            ):
                with torch.no_grad():
                    reconstructions = self(samples_to_log)

                # Store reconstructions for this epoch
                self.fixed_sample_reconstructions[f"epoch_{self.current_epoch}"] = (
                    reconstructions
                )

                # Log visualizations to wandb
                for i, (orig, recon) in enumerate(zip(samples_to_log, reconstructions)):
                    # Visualize original
                    orig_viz = self.reconstruct_and_visualize(
                        orig, f"sample_{i}_original"
                    )
                    self.logger.experiment.log(orig_viz)

                    # Visualize reconstruction
                    recon_viz = self.reconstruct_and_visualize(
                        recon, f"sample_{i}_reconstructed"
                    )
                    self.logger.experiment.log(recon_viz)

        return val_log_dict

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.optimizer_config["lr"],
            betas=tuple(self.optimizer_config["betas"]),
            eps=self.optimizer_config["eps"],
            weight_decay=self.optimizer_config["weight_decay"],
        )

        # Configure scheduler
        if self.scheduler_config["name"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config["T_max"],
                eta_min=self.scheduler_config["eta_min"],
            )
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
