import numpy as np
import torch.nn.functional as F
from dataclasses import asdict
import torch
import pytorch_lightning as pl
from torch import Tensor
from typing import Tuple, Dict, List, Optional
import torch.nn as nn
import math
import wandb

from src.core.config import TransformerExperimentConfig
from src.models.utils import tokens_to_image_dict
from src.models.autoencoder.transformer import Encoder, Decoder
from src.data.inr import INR

demo_inr = INR(up_scale=16)
for param in demo_inr.parameters():
    param.requires_grad = False


class Autoencoder(pl.LightningModule):
    """Autoencoder model with contrastive learning capabilities."""

    def __init__(self, config: TransformerExperimentConfig):
        super().__init__()
        self.save_hyperparameters(asdict(config))
        self.config = config

        # Initialize encoder and decoder
        self.encoder = Encoder(
            max_positions=config.model.max_positions,
            num_layers=config.model.num_layers,
            d_model=config.model.d_model,
            dropout=config.model.dropout,
            num_heads=config.model.num_heads,
            input_dim=config.model.input_dim,
            latent_dim=config.model.latent_dim,
        )

        self.decoder = Decoder(
            max_positions=config.model.max_positions,
            num_layers=config.model.num_layers,
            d_model=config.model.d_model,
            dropout=config.model.dropout,
            num_heads=config.model.num_heads,
            input_dim=config.model.input_dim,
            latent_dim=config.model.latent_dim,
        )

        # Initialize demo INR for visualization
        self.demo_inr = demo_inr.to(self.device)

        # Initialize tracking variables
        self.fixed_val_samples: list[Tensor] | None = None
        self.fixed_train_samples: list[Tensor] | None = None

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

    def _get_grad_norm(self, parameters):
        """Calculate gradient norm for a set of parameters"""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        params_norms = [
            torch.norm(p.grad.detach(), 2) for p in parameters if p.grad is not None
        ]
        return (
            torch.norm(torch.stack(params_norms), 2)
            if params_norms
            else torch.tensor(0.0)
        )

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before optimizer step"""
        # Compute total norm over all parameters
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach(), 2)
                    for p in self.parameters()
                    if p.grad is not None
                ]
            ),
            2,
        )
        self.log("train/grad_norm", total_norm, prog_bar=False, sync_dist=True)

        # Add per-component gradient norms for debugging
        self.log(
            "train/encoder_grad_norm",
            self._get_grad_norm(self.encoder.parameters()),
            prog_bar=False,
        )
        self.log(
            "train/decoder_grad_norm",
            self._get_grad_norm(self.decoder.parameters()),
            prog_bar=False,
        )

    @torch.no_grad()
    def visualize_reconstructions(self, samples, prefix: str, batch_idx: int):
        """Visualize reconstructions and log them."""
        if samples is None:
            print("Fixed samples not initialized.")
            return

        # Process batch
        tokens, masks, positions = samples
        latent, reconstructed = self.forward(tokens, positions)
        reference_checkpoint = self.trainer.val_dataloaders.dataset.get_state_dict(0)

        vis_dict = tokens_to_image_dict(
            reconstructed,  # tokens
            positions,  # pos
            self.demo_inr,
            f"{prefix}/reconstruction",
            self.device,
            reference_checkpoint,
        )

        vis_dict["global_step"] = self.global_step
        self.logger.experiment.log(vis_dict)

    @torch.no_grad()
    def log_latent_distribution(self, latent: Tensor, prefix: str):
        """Log histograms and other visualizations of the latent space"""
        batch_size = latent.shape[0]
        latent_flat = latent.reshape(batch_size, -1)

        # Log histogram of latent values
        self.logger.experiment.log(
            {
                f"{prefix}/latent_histogram": wandb.Histogram(
                    latent_flat.cpu().numpy()
                ),
                f"{prefix}/latent_abs_histogram": wandb.Histogram(
                    torch.abs(latent_flat).cpu().numpy()
                ),
            }
        )

    def on_train_batch_start(self, batch, batch_idx):
        """Log learning rate at the start of each training batch"""
        opt = self.optimizers()
        if opt is not None:
            current_lr = opt.param_groups[0]["lr"]
            self.log("train/lr", current_lr, prog_bar=True)

    def on_train_start(self):
        """Setup fixed validation and training samples for visualization."""
        num_samples = self.config.logging.num_samples_to_visualize

        val_num_samples = (
            num_samples
            if num_samples < len(self.trainer.val_dataloaders.dataset)
            else len(self.trainer.val_dataloaders.dataset)
        )

        train_num_samples = (
            num_samples
            if num_samples < len(self.trainer.train_dataloader.dataset)
            else len(self.trainer.train_dataloader.dataset)
        )

        self.fixed_val_samples = [
            self.trainer.val_dataloaders.dataset[i] for i in range(val_num_samples)
        ]
        self.fixed_val_samples = [
            torch.stack(x).to(self.device) for x in zip(*self.fixed_val_samples)
        ]

        self.fixed_train_samples = [
            self.trainer.train_dataloader.dataset[i] for i in range(train_num_samples)
        ]
        self.fixed_train_samples = [
            torch.stack(x).to(self.device) for x in zip(*self.fixed_train_samples)
        ]

        reference_checkpoint = self.trainer.val_dataloaders.dataset.get_state_dict(0)

        v_tokens, v_masks, v_pos = self.fixed_val_samples
        t_tokens, t_masks, t_pos = self.fixed_train_samples

        v = tokens_to_image_dict(
            v_tokens,  # tokens
            v_pos,  # pos
            self.demo_inr,
            "val/original",
            self.device,
            reference_checkpoint,
        )
        t = tokens_to_image_dict(
            t_tokens,  # tokens
            t_pos,  # pos
            self.demo_inr,
            "train/original",
            self.device,
            reference_checkpoint,
        )

        self.logger.experiment.log(v)
        self.logger.experiment.log(t)

    def forward(
        self,
        input_tokens: Tensor,
        positions: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            input_tokens: Input token sequence [batch_size, seq_len, input_dim]
            positions: Position encodings [batch_size, seq_len, n_pos_dims]
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            latent: Encoded latent representations
            reconstructed: Reconstructed token sequence
        """
        latent = self.encoder(input_tokens, positions, attention_mask)
        reconstructed = self.decoder(latent, positions, attention_mask)
        return latent, reconstructed

    def compute_loss(
        self,
        latent: Tensor,
        reconstructed: Tensor,
        original_tokens: Tensor,
        prefix: str = "train",
        beta: float = 1e-5,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute VAE loss optimized for latent diffusion with improved stability and logging
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, original_tokens)

        # Reshape and normalize latent
        batch_size = latent.shape[0]
        latent_flat = latent.reshape(batch_size, -1)

        # KL loss with improved numerical stability
        kl_loss = 0.5 * torch.mean(
            torch.sum(latent_flat.pow(2), dim=1)  # E[z^2] term
            + math.log(2 * math.pi)  # Constant term, moved outside sum
            + 1  # log(σ²) term for standard normal
        )

        # Scale losses
        # total_loss = recon_loss + beta * kl_loss
        total_loss = recon_loss

        # Detailed logging dictionary
        loss_dict = {
            f"{prefix}/loss_recon": recon_loss,
            f"{prefix}/loss_kl": kl_loss,
            f"{prefix}/loss": total_loss,
        }
        return total_loss, loss_dict

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        original_tokens, original_masks, original_positions = batch
        latent, reconstructed_tokens = self.forward(original_tokens, original_positions)

        # Compute loss
        loss, log_dict = self.compute_loss(
            latent,
            reconstructed_tokens,
            original_tokens,
            prefix="train",
        )

        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            self.log_latent_distribution(latent, "train")

            if self.current_epoch % self.config.logging.sample_every_n_epochs == 0:
                self.visualize_reconstructions(
                    self.fixed_train_samples, "train", batch_idx
                )

        return loss

    def validation_step(self, batch, batch_idx: int) -> dict[str, Tensor]:
        original_tokens, original_masks, original_positions = batch
        latent, reconstructed_tokens = self.forward(original_tokens, original_positions)

        # Compute loss
        val_loss, val_log_dict = self.compute_loss(
            latent,
            reconstructed_tokens,
            original_tokens,
            prefix="val",
        )

        # Log validation metrics
        self.log_dict(val_log_dict, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            self.log_latent_distribution(latent, "val")

            if self.current_epoch % self.config.logging.sample_every_n_epochs == 0:
                self.visualize_reconstructions(self.fixed_val_samples, "val", batch_idx)

        return val_log_dict

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
