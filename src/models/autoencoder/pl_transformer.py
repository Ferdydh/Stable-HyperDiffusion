import copy
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from torch import Tensor
from typing import Tuple, Dict, List, Optional
import torch.nn as nn
import math

from src.data.utils import tokens_to_weights, weights_to_tokens
from src.models.autoencoder.losses import GammaContrastReconLoss
from src.models.autoencoder.transformer import Encoder, Decoder, ProjectionHead
from src.data.inr import INR
from src.models.utils import (
    create_reconstruction_visualizations_with_state_dict as create_reconstruction_visualizations,
)


def transform(tokens, masks, positions):
    """
    Apply augmentations to the input data.
    """

    return tokens, masks, positions, tokens, masks, positions


class Autoencoder(pl.LightningModule):
    """Autoencoder model with contrastive learning capabilities."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)

        # Default configuration
        default_config = OmegaConf.create(
            {
                "model": {
                    "max_positions": [100, 10, 40],
                    "num_layers": 8,
                    "d_model": 1024,
                    "dropout": 0.0,
                    "window_size": 32,
                    "num_heads": 8,
                    "input_dim": 33,
                    "n_tokens": 65,
                    "latent_dim": 128,
                    "projection_dim": 128,
                },
                "training": {
                    "gamma": 0.5,
                    "reduction": "mean",
                    "temperature": 0.1,
                    "contrast": "simclr",
                    "z_var_penalty": 0.0,
                },
                "data": {
                    "batch_size": 64,
                },
                "logging": {
                    "num_samples_to_visualize": 8,
                    "log_every_n_steps": 100,
                    "sample_every_n_epochs": 1,
                },
            }
        )

        # Merge configurations
        self.config = OmegaConf.merge(default_config, config)
        model_config = self.config.model

        # Initialize encoder and decoder
        self.encoder = Encoder(
            max_positions=model_config.max_positions,
            num_layers=model_config.num_layers,
            d_model=model_config.d_model,
            dropout=model_config.dropout,
            window_size=model_config.window_size,
            num_heads=model_config.num_heads,
            input_dim=model_config.input_dim,
            latent_dim=model_config.latent_dim,
        )

        self.decoder = Decoder(
            max_positions=model_config.max_positions,
            num_layers=model_config.num_layers,
            d_model=model_config.d_model,
            dropout=model_config.dropout,
            window_size=model_config.window_size,
            num_heads=model_config.num_heads,
            input_dim=model_config.input_dim,
            latent_dim=model_config.latent_dim,
        )

        # Initialize projection head
        self.projection_head = ProjectionHead(
            d_model=model_config.latent_dim,
            n_tokens=model_config.n_tokens,
            output_dim=model_config.projection_dim,
        )

        # Initialize loss function
        self.criterion = GammaContrastReconLoss(
            gamma=self.config.training.gamma,
            reduction=self.config.training.reduction,
            batch_size=self.config.data.batch_size,
            temperature=self.config.training.temperature,
            contrast=self.config.training.contrast,
            z_var_penalty=self.config.training.z_var_penalty,
        )

        # Initialize demo INR for visualization
        self.demo_inr = INR(up_scale=16).to(self.device)

        # Initialize tracking variables
        self.fixed_val_samples: Optional[List[Tensor]] = None
        self.fixed_train_samples: Optional[List[Tensor]] = None
        self.fixed_sample_reconstructions: Dict[str, List[Tensor]] = {}
        self.best_val_loss: float = float("inf")

        # Store optimizer and scheduler config
        self.optimizer_config = config.optimizer
        self.scheduler_config = config.scheduler

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights with specific distributions."""

        def init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.apply(init_weights)

        # Special initialization for projection layers
        num_layers = self.config.model.num_layers
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    def forward(
        self,
        input_tokens: Tensor,
        positions: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            input_tokens: Input token sequence [batch_size, seq_len, input_dim]
            positions: Position encodings [batch_size, seq_len, n_pos_dims]
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            latent: Encoded latent representations
            reconstructed: Reconstructed token sequence
            projected: Projected latent representations
        """
        latent = self.encoder(input_tokens, positions, attention_mask)
        projected = self.projection_head(latent)
        reconstructed = self.decoder(latent, positions, attention_mask)
        return latent, reconstructed, projected

    def compute_loss(
        self,
        transform_output_i: Tuple[Tensor, Tensor, Tensor],
        transform_output_j: Tuple[Tensor, Tensor, Tensor],
        input_tokens_i: Tensor,
        input_tokens_j: Tensor,
        masks_i: Tensor,
        masks_j: Tensor,
        prefix: str = "train",
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute contrastive and reconstruction loss."""
        latent_i, recon_i, projected_i = transform_output_i
        latent_j, recon_j, projected_j = transform_output_j

        tokens = torch.cat([input_tokens_i, input_tokens_j], dim=0)
        reconstructions = torch.cat([recon_i, recon_j], dim=0)
        masks = torch.cat([masks_i, masks_j], dim=0)

        loss = self.criterion(
            z_i=projected_i, z_j=projected_j, y=reconstructions, t=tokens, m=masks
        )

        return loss, {f"{prefix}/loss": loss}

    def on_train_start(self):
        """Setup fixed validation and training samples for visualization."""
        num_samples = self.config.logging.num_samples_to_visualize
        val_batch = next(iter(self.trainer.val_dataloaders))
        self.fixed_val_samples = copy.deepcopy(val_batch[:num_samples])
        train_batch = next(iter(self.trainer.train_dataloader))
        self.fixed_train_samples = copy.deepcopy(train_batch[:num_samples])

    def _process_batch(self, batch: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert batch to required tensor format."""
        tokens, masks, positions = zip(
            *[weights_to_tokens(b, tokensize=0, device=self.device) for b in batch]
        )
        return (
            torch.stack(tokens).to(self.device),
            torch.stack(masks).to(self.device),
            torch.stack(positions).to(self.device).to(torch.int32),
        )

    def _compute_gradient_norm(self) -> float:
        """Compute total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5

    def _log_metrics(self, metrics: Dict[str, Tensor], batch_idx: int, prefix: str):
        """Log metrics and gradient norm if needed."""
        self.log_dict(
            metrics,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.config.data.batch_size,
        )

        if prefix == "train" and batch_idx % self.config.logging.log_every_n_steps == 0:
            grad_norm = self._compute_gradient_norm()
            self.log(
                "train/grad_norm",
                grad_norm,
                prog_bar=False,
                sync_dist=True,
                batch_size=self.config.data.batch_size,
            )

    def visualize_reconstructions(
        self, samples: List[Tensor], prefix: str, batch_idx: int
    ):
        """Visualize reconstructions and log them."""
        if not samples or batch_idx % self.config.logging.log_every_n_steps != 0:
            return

        with torch.no_grad():
            tokens, masks, positions = self._process_batch(samples)
            latent, reconstructed, _ = self.forward(tokens, positions)

            reconstructions = [
                tokens_to_weights(t, p, samples[0])
                for t, p in zip(reconstructed, positions)
            ]

            step_key = f"{prefix}_step_{self.global_step}"
            self.fixed_sample_reconstructions[step_key] = reconstructions

            vis_dict = create_reconstruction_visualizations(
                samples,
                reconstructions,
                self.demo_inr,
                prefix,
                batch_idx,
                self.global_step,
                is_fixed=True,
            )
            vis_dict["global_step"] = self.global_step
            self.logger.experiment.log(vis_dict)

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """Execute a single training step."""
        # Process batch
        tokens, masks, positions = self._process_batch(batch)

        # Get transformed versions (placeholder for now)
        tokens_i, masks_i, positions_i, tokens_j, masks_j, positions_j = transform(
            tokens, masks, positions
        )

        # Forward pass
        transform_output_i = self.forward(tokens_i, positions_i)
        transform_output_j = self.forward(tokens_j, positions_j)

        # Compute loss
        loss, log_dict = self.compute_loss(
            transform_output_i,
            transform_output_j,
            tokens_i,
            tokens_j,
            masks_i,
            masks_j,
            prefix="train",
        )

        # Log metrics
        self._log_metrics(log_dict, batch_idx, "train")

        # Visualize reconstructions
        self.visualize_reconstructions(self.fixed_train_samples, "train", batch_idx)

        return loss

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """Execute a single validation step."""
        # Process batch
        tokens, masks, positions = self._process_batch(batch)

        # Get transformed versions (placeholder for now)
        tokens_i, masks_i, positions_i, tokens_j, masks_j, positions_j = transform(
            tokens, masks, positions
        )

        # Forward pass
        transform_output_i = self.forward(tokens_i, positions_i)
        transform_output_j = self.forward(tokens_j, positions_j)

        # Compute loss
        loss, log_dict = self.compute_loss(
            transform_output_i,
            transform_output_j,
            tokens_i,
            tokens_j,
            masks_i,
            masks_j,
            prefix="val",
        )

        # Log metrics
        self._log_metrics(log_dict, batch_idx, "val")

        # Visualize reconstructions
        if (
            batch_idx == 0
            and self.current_epoch % self.config.logging.sample_every_n_epochs == 0
        ):
            self.visualize_reconstructions(self.fixed_val_samples, "val", batch_idx)

        return log_dict

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Configure optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config.lr,
            betas=tuple(self.optimizer_config.betas),
            eps=self.optimizer_config.eps,
            weight_decay=self.optimizer_config.weight_decay,
        )

        # Configure scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.scheduler_config.T_max,
            eta_min=self.scheduler_config.eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
