from transformers import get_linear_schedule_with_warmup
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

    def on_after_optimizer_step(self, optimizer):
        return self._log_grad_norm(optimizer, prefix="after")

    def on_before_optimizer_step(self, optimizer):
        return self._log_grad_norm(optimizer, prefix="before")

    def _log_grad_norm(self, optimizer, prefix: str):
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
        self.log(f"{prefix}/grad_norm", total_norm, prog_bar=False, sync_dist=True)

        # Add per-component gradient norms for debugging
        self.log(
            f"{prefix}/encoder_grad_norm",
            self._get_grad_norm(self.encoder.parameters()),
            prog_bar=False,
        )
        self.log(
            f"{prefix}/decoder_grad_norm",
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
        reconstructed, z, mu, logvar = self.forward(tokens, positions)
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
    def log_latent_distribution(self, mu: Tensor, logvar: Tensor, prefix: str):
        batch_size = mu.shape[0]
        mu_flat = mu.reshape(batch_size, -1)

        # Most commonly analyzed metrics
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        self.logger.experiment.log(
            {
                # KL divergence metrics - key for detecting posterior collapse
                f"{prefix}/mean_kl_per_dim": kl_per_dim.mean().item(),
                f"{prefix}/kl_histogram": wandb.Histogram(kl_per_dim.cpu().numpy()),
                # Mean statistics - to check if latent space is centered
                f"{prefix}/mean_mu": mu_flat.mean().item(),
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

    #@torch.compile
    def forward(
        self,
        input_tokens: Tensor,
        positions: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass through the VAE.

        Args:
            input_tokens: Input token sequence [batch_size, seq_len, input_dim]
            positions: Position encodings [batch_size, seq_len, n_pos_dims]
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            reconstructed: Reconstructed token sequence
            z: Sampled latent vectors
            mu: Mean of the latent Gaussian
            logvar: Log variance of the latent Gaussian
        """
        # Get latent distribution parameters and sample
        z, mu, logvar = self.encoder(input_tokens, positions, attention_mask)

        # Reconstruct from sampled latent
        reconstructed = self.decoder(z, positions, attention_mask)

        return reconstructed, z, mu, logvar

    def compute_loss(
        self,
        reconstructed: Tensor,
        original: Tensor,
        mask: Tensor,
        mu: Tensor,
        logvar: Tensor,
        prefix: str = "train"
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute VAE loss optimized for latent diffusion with improved stability and logging
        """
        
        if self.config.model.use_mask:
            reconstructed = mask * reconstructed
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, original)

        # KL divergence loss
        # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        )

        # Total loss with β weighting
        # Recon loss goal is 1e-6
        total_loss = recon_loss * self.config.model.recon_scale + self.config.model.beta * kl_loss

        # Detailed logging dictionary
        loss_dict = {
            f"{prefix}/loss_recon": recon_loss,
            f"{prefix}/loss_kl": kl_loss,
            f"{prefix}/loss": total_loss,
        }
        return total_loss, loss_dict

    #@torch.compile
    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        original_tokens, original_masks, original_positions = batch
        original_tokens = original_tokens.to(self.device)
        original_masks = original_masks.to(self.device)
        original_positions = original_positions.to(self.device)
        reconstructed, z, mu, logvar = self.forward(original_tokens, original_positions)

        # Compute loss
        loss, log_dict = self.compute_loss(
            reconstructed,
            original_tokens,
            original_masks,
            mu,
            logvar,
            prefix="train",
        )

        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            self.log_latent_distribution(mu, logvar, "train")

            if self.current_epoch % self.config.logging.sample_every_n_epochs == 0:
                self.visualize_reconstructions(
                    self.fixed_train_samples, "train", batch_idx
                )

        return loss

    #@torch.compile
    def validation_step(self, batch, batch_idx: int) -> dict[str, Tensor]:
        original_tokens, original_masks, original_positions = batch
        original_tokens = original_tokens.to(self.device)
        original_masks = original_masks.to(self.device)
        original_positions = original_positions.to(self.device)
        reconstructed, z, mu, logvar = self.forward(original_tokens, original_positions)

        # Compute loss
        loss, log_dict = self.compute_loss(
            reconstructed,
            original_tokens,
            original_masks,
            mu,
            logvar,
            prefix="val",
        )

        # Log validation metrics
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            self.log_latent_distribution(mu, logvar, "val")

            if self.current_epoch % self.config.logging.sample_every_n_epochs == 0:
                self.visualize_reconstructions(self.fixed_val_samples, "val", batch_idx)

        return log_dict

    def configure_optimizers(self):
        # AdamW optimizer with explicit defaults
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.optimizer.lr,  # typically 2e-4 or 5e-4
            betas=tuple(self.config.optimizer.betas),  # typically (0.9, 0.999)
            eps=self.config.optimizer.eps,  # typically 1e-8
            weight_decay=self.config.optimizer.weight_decay,  # typically 0.01
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.config.scheduler.warmup_ratio)

        # Linear warmup scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Return in Lightning format
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Important: call scheduler every step, not epoch
                "frequency": 1,
                "monitor": self.config.checkpoint.monitor,
            },
        }
