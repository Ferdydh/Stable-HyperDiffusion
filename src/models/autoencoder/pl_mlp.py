from dataclasses import asdict
import torch
import pytorch_lightning as pl
from torch import Tensor
from typing import Tuple
import torch.nn as nn

from src.core.config import MLPExperimentConfig, MLPModelConfig
from src.data.inr import INR
from src.models.utils import log_original_image, log_reconstructed_image


class Encoder(nn.Module):
    def __init__(self, config: MLPModelConfig):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc3 = nn.Linear(config.hidden_dim, config.z_dim)
        self.activation_func = config.activation()

    def forward(self, x):
        x = self.activation_func(self.fc1(x))
        x = self.activation_func(self.fc2(x))
        z = self.fc3(x)
        return z


class Decoder(nn.Module):
    def __init__(self, config: MLPModelConfig):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(config.z_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc3 = nn.Linear(config.hidden_dim, config.output_dim)
        self.activation_func = config.activation()

    def forward(
        self,
        z: Tensor,
    ) -> Tensor:
        z = self.activation_func(self.fc1(z))
        z = self.activation_func(self.fc2(z))
        x_reconstructed = self.fc3(z)
        return x_reconstructed


class Autoencoder(pl.LightningModule):
    def __init__(self, config: MLPExperimentConfig):
        super().__init__()
        self.save_hyperparameters(asdict(config))
        self.config = config

        # Initialize encoder and decoder
        self.encoder = Encoder(config.model)
        self.decoder = Decoder(config.model)

        # Initialize fixed validation and training samples
        self.fixed_val_samples: list[Tensor] | None = None
        self.fixed_train_samples: list[Tensor] | None = None

        # Initialize loss function
        self.loss_func = nn.MSELoss()

        # Initialize quality metrics
        self.best_val_loss = float("inf")

        # Create demo INR for visualization
        self.demo_inr = INR(up_scale=16).to(self.device)

    def on_train_start(self):
        """Setup fixed validation and training samples for visualization."""
        num_samples = self.config.logging.num_samples_to_visualize

        self.fixed_val_samples = torch.stack(
            [self.trainer.val_dataloaders.dataset[i] for i in range(num_samples)]
        ).to(self.device)

        self.fixed_train_samples = torch.stack(
            [self.trainer.train_dataloader.dataset[i] for i in range(num_samples)]
        ).to(self.device)

        v = log_original_image(self.fixed_val_samples, self.demo_inr, "val")
        t = log_original_image(self.fixed_train_samples, self.demo_inr, "train")

        self.logger.experiment.log(v)
        self.logger.experiment.log(t)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, input: Tensor) -> Tensor:
        z = self.encode(input)
        dec = self.decode(z)
        return dec

    def compute_loss(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        prefix: str = "train",
    ) -> Tuple[Tensor, dict[str, Tensor]]:
        recon_loss = self.loss_func(reconstructions, inputs)
        return recon_loss, {f"{prefix}/loss": recon_loss}

    def visualize_reconstructions(self, samples: Tensor, prefix: str, batch_idx: int):
        """Helper method to visualize fixed sample reconstructions during training or validation."""
        if samples is None:
            print("Fixed samples not initialized.")
            return

        with torch.no_grad():
            reconstructions = self.forward(samples)

        # Create and log visualizations
        vis_dict = log_reconstructed_image(
            reconstructions,
            self.demo_inr,
            prefix,
        )

        # Add step to wandb log
        vis_dict["global_step"] = self.global_step
        self.logger.experiment.log(vis_dict)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        # Forward pass
        reconstructions = self.forward(batch)
        loss, log_dict = self.compute_loss(batch, reconstructions, prefix="train")

        self.log_dict(log_dict, prog_bar=True)

        # Log gradient norm
        if batch_idx % self.config.trainer.log_every_n_steps == 0:
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            self.log("train/grad_norm", total_norm, prog_bar=False, sync_dist=True)

            # Visualize both fixed samples and current batch
            self.visualize_reconstructions(self.fixed_train_samples, "train", batch_idx)

        return loss

    def validation_step(self, batch, batch_idx: int) -> dict[str, Tensor]:
        reconstructions = self.forward(batch)
        val_loss, val_log_dict = self.compute_loss(batch, reconstructions, prefix="val")

        # Log validation metrics
        self.log_dict(val_log_dict, prog_bar=True, batch_size=batch.shape[0])

        # Visualize both fixed samples and current batch
        if (
            batch_idx == 0
            and self.current_epoch % self.config.logging.sample_every_n_epochs == 0
        ):
            self.visualize_reconstructions(self.fixed_val_samples, "val", batch_idx)

        return val_loss

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.optimizer.lr,
            betas=tuple(self.config.optimizer.betas),
            eps=self.config.optimizer.eps,
            weight_decay=self.config.optimizer.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.scheduler.T_max,
            eta_min=self.config.scheduler.eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.config.checkpoint.monitor,
            },
        }
