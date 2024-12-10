from matplotlib import pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import wandb
from typing import Dict, Any

from core.utils import plot_image
from models.autoencoder import load_weights_into_inr
from models.inr import INR

from .gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=False)

    def forward(self, x):
        freqs = x[:, None] * self.weights[None, :] * 2 * np.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return fouriered


class DiffusionModel(nn.Module):
    """Simple MLP-based model for processing INR weights."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.input_dim = config["model"]["input_dim"]
        self.hidden_dim = config["model"]["hidden_dim"]
        self.time_embed_dim = config["model"]["time_embed_dim"]

        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.ReLU(),
        )

        # Main network
        self.net = nn.Sequential(
            nn.Linear(self.input_dim + self.time_embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config["model"]["dropout"]),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config["model"]["dropout"]),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, x, t):
        t_embed = self.time_embed(t)
        h = torch.cat((x, t_embed), dim=-1)
        return self.net(h)


class INRDiffusion(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Create model
        self.model = DiffusionModel(config)

        # Create diffusion process
        betas = torch.linspace(
            config["diffusion"]["beta_start"],
            config["diffusion"]["beta_end"],
            config["diffusion"]["timesteps"],
        )

        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType[config["diffusion"]["model_mean_type"]],
            model_var_type=ModelVarType[config["diffusion"]["model_var_type"]],
            loss_type=LossType[config["diffusion"]["loss_type"]],
            diff_pl_module=self,
        )

        # Store demo INR for visualization
        self.demo_inr = INR(up_scale=16)
        self.demo_inr = self.demo_inr.to(self.device)

        # Initialize fixed validation samples
        self.fixed_samples = None

    def setup(self, stage: str | None = None):
        """Setup fixed samples for visualization."""
        if stage == "fit":
            try:
                num_samples = self.config["training"].get("num_samples_to_visualize", 4)

                if (
                    hasattr(self.trainer, "val_dataloaders")
                    and self.trainer.val_dataloaders is not None
                    and self.fixed_samples is None
                ):
                    val_batch = next(iter(self.trainer.val_dataloaders[0]))
                    self.fixed_samples = val_batch[:num_samples].clone()

            except Exception as e:
                print(f"Warning: Could not setup fixed samples: {e}")
                self.fixed_samples = None

    def training_step(self, batch, batch_idx):
        # Sample timesteps
        b = batch.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (b,), device=self.device)

        # Calculate loss
        loss_dict = self.diffusion.training_losses(
            self.model,
            batch * self.config["training"]["normalization_factor"],
            t,
            None,  # MLP kwargs not needed for 2D case
            self.logger,
        )

        loss = loss_dict["loss"].mean()
        self.log("train/loss", loss, prog_bar=True)

        # Visualize samples periodically
        if batch_idx % self.config["training"]["eval_interval"] == 0:
            self.visualize_samples("train", batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.visualize_samples("val", batch_idx)

    def visualize_samples(self, prefix: str, batch_idx: int):
        """Generate and visualize samples."""
        if self.demo_inr is None:
            return

        # Generate samples
        samples = self.diffusion.ddim_sample_loop(
            self.model,
            (
                self.config["training"].get("num_samples_to_visualize", 4),
                self.config["model"]["input_dim"],
            ),
            device=self.device,
        )

        # Denormalize samples
        samples = samples / self.config["training"]["normalization_factor"]

        # Create visualizations
        vis_dict = {}
        for i, sample in enumerate(samples):
            # Generate figure using the provided visualization function
            fig = plot_image(load_weights_into_inr(sample, self.demo_inr), self.device)
            vis_dict[f"{prefix}/sample_{i}"] = wandb.Image(fig)
            plt.close(fig)

        # If we have fixed samples, also visualize them
        if self.fixed_samples is not None:
            for i, fixed in enumerate(self.fixed_samples):
                fig = plot_image(
                    load_weights_into_inr(fixed, self.demo_inr), self.device
                )
                vis_dict[f"{prefix}/fixed_{i}"] = wandb.Image(fig)
                plt.close(fig)

        # Add step to wandb log
        vis_dict["global_step"] = self.global_step
        self.logger.experiment.log(vis_dict)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["training"]["lr"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        return optimizer

    @torch.no_grad()
    def sample(self, batch_size: int = 1):
        """Generate samples."""
        samples = self.diffusion.ddim_sample_loop(
            self.model,
            (batch_size, self.config["model"]["input_dim"]),
            device=self.device,
        )
        return samples / self.config["training"]["normalization_factor"]
