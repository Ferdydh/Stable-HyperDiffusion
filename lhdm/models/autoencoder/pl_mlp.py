import torch
import pytorch_lightning as pl
from torch import Tensor
from typing import Tuple
from jaxtyping import Float
from typeguard import typechecked
import torch.nn as nn

from models.inr import INR
from models.utils import create_reconstruction_visualizations, get_activation


class Encoder(nn.Module):
    @typechecked
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        z_dim: int,
        activation: str = "relu",
        **kwargs,
    ):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, z_dim)
        self.activation_func = get_activation(activation)

    @typechecked
    def forward(
        self, x: Float[Tensor, "batch input_dim"]
    ) -> Float[Tensor, "batch z_dim"]:
        x = self.activation_func(self.fc1(x))
        x = self.activation_func(self.fc2(x))
        z = self.fc3(x)
        return z


class Decoder(nn.Module):
    @typechecked
    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "relu",
        **kwargs,
    ):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation_func = get_activation(activation)

    @typechecked
    def forward(
        self, z: Float[Tensor, "batch z_dim"]
    ) -> Float[Tensor, "batch output_dim"]:
        z = self.activation_func(self.fc1(z))
        z = self.activation_func(self.fc2(z))
        x_reconstructed = self.fc3(z)
        return x_reconstructed


class Autoencoder(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Initialize encoder and decoder
        self.encoder = Encoder(**config["model"])
        self.decoder = Decoder(**config["model"])

        # Initialize fixed validation and training samples
        self.fixed_val_samples: list[Tensor] | None = None
        self.fixed_train_samples: list[Tensor] | None = None
        self.fixed_sample_reconstructions: dict[str, list[Tensor]] = {}

        # Store optimizer and scheduler config
        self.optimizer_config = config["optimizer"]
        self.scheduler_config = config["scheduler"]

        # Initialize loss function
        self.loss_func = nn.MSELoss()

        # Initialize quality metrics
        self.best_val_loss = float("inf")

        # Create demo INR for visualization
        self.demo_inr = INR(up_scale=16)
        # Move demo INR to the same device as the model
        self.demo_inr = self.demo_inr.to(self.device)

    def setup(self, stage: str | None = None):
        """Setup fixed validation and training samples for tracking reconstruction progress."""
        if stage == "fit":
            try:
                num_samples = self.config["logging"]["num_samples_to_visualize"]

                # Setup validation samples
                if (
                    hasattr(self.trainer, "val_dataloaders")
                    and self.trainer.val_dataloaders is not None
                    and self.fixed_val_samples is None
                ):
                    val_batch = next(iter(self.trainer.val_dataloaders[0]))
                    self.fixed_val_samples = val_batch[:num_samples].clone()

                # Setup training samples
                if (
                    hasattr(self.trainer, "train_dataloader")
                    and self.trainer.train_dataloader is not None
                    and self.fixed_train_samples is None
                ):
                    train_batch = next(iter(self.trainer.train_dataloader()))
                    self.fixed_train_samples = train_batch[:num_samples].clone()

            except Exception as e:
                print(f"Warning: Could not setup fixed samples: {e}")
                self.fixed_val_samples = None
                self.fixed_train_samples = None

        print(self.fixed_val_samples)
        print(self.fixed_train_samples)

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
        recon_loss = self.loss_func(reconstructions, inputs)
        return recon_loss, {f"{prefix}/loss": recon_loss}

    def visualize_batch(self, batch: Tensor, prefix: str, batch_idx: int):
        """Visualize a batch of samples during training or validation."""
        if batch_idx % self.config["logging"]["log_every_n_steps"] == 0:
            with torch.no_grad():
                reconstructions = self(batch)

            # Log visualizations for a subset of the batch
            num_samples = min(
                self.config["logging"]["num_samples_to_visualize"], batch.shape[0]
            )  # Visualize up to num_samples_to_visualize
            vis_dict = create_reconstruction_visualizations(
                batch[:num_samples],
                reconstructions[:num_samples],
                self.demo_inr,
                prefix,
                batch_idx,
                self.global_step,
                is_fixed=False,
            )

            # Add step to wandb log
            vis_dict["global_step"] = self.global_step
            self.logger.experiment.log(vis_dict)

    def visualize_reconstructions(self, samples: Tensor, prefix: str, batch_idx: int):
        """Helper method to visualize fixed sample reconstructions during training or validation."""
        if (
            samples is not None
            and batch_idx % self.config["logging"]["log_every_n_steps"] == 0
        ):
            with torch.no_grad():
                reconstructions = self(samples)

            # Store reconstructions for this step
            step_key = f"{prefix}_step_{self.global_step}"
            self.fixed_sample_reconstructions[step_key] = reconstructions

            # Create and log visualizations
            vis_dict = create_reconstruction_visualizations(
                samples,
                reconstructions,
                self.demo_inr,
                prefix,
                batch_idx,
                self.global_step,
                is_fixed=True,
            )

            # Add step to wandb log
            vis_dict["global_step"] = self.global_step
            self.logger.experiment.log(vis_dict)

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

            # Visualize both fixed samples and current batch
            self.visualize_reconstructions(self.fixed_train_samples, "train", batch_idx)
            self.visualize_batch(batch, "train_batch", batch_idx)

        return loss

    @typechecked
    def validation_step(
        self, batch: Float[Tensor, "batch feature_dim"], batch_idx: int
    ) -> dict[str, Tensor]:
        reconstructions = self(batch)
        val_loss, val_log_dict = self.compute_loss(batch, reconstructions, prefix="val")

        # Log validation metrics
        self.log_dict(val_log_dict, prog_bar=True, sync_dist=True)

        # Visualize both fixed samples and current batch
        if (
            batch_idx == 0
            and self.current_epoch % self.config["logging"]["sample_every_n_epochs"]
            == 0
        ):
            self.visualize_reconstructions(self.fixed_val_samples, "val", batch_idx)
            self.visualize_batch(batch, "val_batch", batch_idx)

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
