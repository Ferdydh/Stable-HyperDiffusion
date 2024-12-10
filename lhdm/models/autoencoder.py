from matplotlib import pyplot as plt
import torch
import pytorch_lightning as pl
import wandb
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
from jaxtyping import Float
from typeguard import typechecked
import torch.nn as nn
import numpy as np

from models.inr import INR


def plot_image(
    mlp_model: INR, device: torch.device
) -> plt.Figure:  # Updated return type hint
    resolution = 28
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    grid_x, grid_y = np.meshgrid(x, y)

    inputs = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = mlp_model(inputs_tensor).cpu()

    image = outputs.reshape(resolution, resolution)
    image_local = image.numpy()

    fig, ax = plt.subplots()
    ax.imshow(image_local, cmap="gray", extent=(-1, 1, -1, 1))
    plt.axis("off")
    return fig, image


def load_weights_into_inr(weights: Tensor, inr_model: INR) -> INR:
    """Helper function to load weights into INR model."""
    state_dict = {}
    start_idx = 0
    for key, param in inr_model.state_dict().items():
        param_size = param.numel()
        param_data = weights[start_idx : start_idx + param_size].reshape(param.shape)
        state_dict[key] = param_data
        start_idx += param_size
    inr_model.load_state_dict(state_dict)
    return inr_model


def create_reconstruction_visualizations(
    originals: Tensor,
    reconstructions: Tensor,
    inr_model: INR,
    prefix: str,
    batch_idx: int,
    global_step: int,
    is_fixed: bool = False,
) -> dict:
    """Create visualization grid for original-reconstruction pairs."""
    result_dict = {}

    # Create visualizations for each pair
    for i, (orig, recon) in enumerate(zip(originals, reconstructions)):

        # Generate figures
        original_fig, og_image = plot_image(load_weights_into_inr(orig, inr_model), orig.device)
        recon_fig, recon_image = plot_image(load_weights_into_inr(recon, inr_model), recon.device)

        # Calculate MSE between original (weights / reconstructed images) and predicted (weights / reconstructed images)
        weight_mse = nn.functional.mse_loss(orig, recon).item()
        image_mse = nn.functional.mse_loss(og_image, recon_image).item()

        # Add to result dictionary with unique keys
        result_dict[f"{prefix}/original_{i}"] = wandb.Image(original_fig)
        result_dict[f"{prefix}/reconstruction_{i}"] = wandb.Image(
            recon_fig,
            caption=f"Image MSE: {image_mse:.6f}, Weight MSE: {weight_mse:.6f}"
        )

        # Close figures
        plt.close(original_fig)
        plt.close(recon_fig)

    return result_dict


@typechecked
def get_activation(activation_name: str) -> nn.Module:
    """Helper function to map activation names to PyTorch functions."""
    activations = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU
    }
    return activations.get(activation_name.lower(), nn.ReLU)()

@typechecked
def get_loss_function(loss_name: str) -> nn.Module:
    """Helper function to map loss names to PyTorch functions."""
    loss = {
        "mae": nn.L1Loss,
        "mse": nn.MSELoss
    }
    return loss.get(loss_name.lower(), nn.L1Loss)()


class Encoder(nn.Module):
    @typechecked
    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int, activation: str = "relu", **kwargs):
        super(Encoder, self).__init__()

        self.activation_func = get_activation(activation)
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim), 
            nn.BatchNorm1d(input_dim),
            self.activation_func,
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self.activation_func,
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            self.activation_func,          
            nn.Linear(512, z_dim),
        )

    @typechecked
    def forward(
        self, x: Float[Tensor, "batch input_dim"]
    ) -> Float[Tensor, "batch z_dim"]:
        return self.model(x)


class Decoder(nn.Module):
    @typechecked
    def __init__(self, z_dim: int, hidden_dim: int, output_dim: int, activation: str = "relu", **kwargs):
        super(Decoder, self).__init__()
        self.activation_func = get_activation(activation)
        self.model = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.BatchNorm1d(512),
            self.activation_func,
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self.activation_func,
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            self.activation_func,
            nn.Linear(output_dim, output_dim)
        )

    @typechecked
    def forward(
        self, z: Float[Tensor, "batch z_dim"]
    ) -> Float[Tensor, "batch output_dim"]:
        return self.model(z)


class Autoencoder(pl.LightningModule):
    @typechecked
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
        self.loss_func = get_loss_function(config["model"]["loss"])

        # Initialize quality metrics
        self.best_val_loss = float("inf")

        # Create demo INR for visualization
        self.demo_inr = INR(up_scale=16)
        # Move demo INR to the same device as the model
        self.demo_inr = self.demo_inr.to(self.device)

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
        recon_loss = self.loss_func(reconstructions, inputs)
        return recon_loss, {f"{prefix}/loss": recon_loss}

    def visualize_batch(self, batch: Tensor, prefix: str, batch_idx: int):
        """Visualize a batch of samples during training or validation."""
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

    @typechecked
    def training_step(
        self, batch: Float[Tensor, "batch feature_dim"], batch_idx: int
    ) -> Tensor:
        # Forward pass
        reconstructions = self(batch)
        loss, log_dict = self.compute_loss(batch, reconstructions, prefix="train")

        # Logging
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        # Log gradient norm and weights
        if self.current_epoch % self.config["logging"]["sample_every_n_epochs"] == 0:
            total_norm = 0.0
            result_dict = {}
            
            for name, param in self.encoder.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    result_dict[f"train/gradients/encoder/{name}"] = wandb.Histogram(param.grad.data.cpu().detach().numpy())
                result_dict[f"train/weights/encoder/{name}"] = wandb.Histogram(param.cpu().detach().numpy())
            for name, param in self.encoder.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    result_dict[f"train/gradients/decoder/{name}"] = wandb.Histogram(param.grad.data.cpu().detach().numpy())
                result_dict[f"train/weights/decoder/{name}"] = wandb.Histogram(param.cpu().detach().numpy())
            result_dict["global_step"] = self.global_step
            self.logger.experiment.log(result_dict)
            total_norm = total_norm**0.5
            self.log("train/grad_norm", total_norm, prog_bar=False, sync_dist=True)
            
            # Visualize both fixed samples and current batch
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
        if self.current_epoch % self.config["logging"]["sample_every_n_epochs"] == 0:
            self.visualize_batch(batch, "val_batch", batch_idx)

        return val_log_dict

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
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
        elif self.scheduler_config["name"] == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_config["factor"],
                patience=self.scheduler_config["patience"],
                min_lr=self.scheduler_config["min_lr"],
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
