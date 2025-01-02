from dataclasses import asdict
import torch
import pytorch_lightning as pl
from torch import Tensor
from typing import Tuple, Dict, List, Optional
import torch.nn as nn
import math

from src.core.config import TransformerExperimentConfig
from src.models.utils import (
    tokens_to_image_dict,
)
from src.models.autoencoder.losses import GammaContrastReconLoss
from src.models.autoencoder.transformer import Encoder, Decoder, ProjectionHead
from src.data.inr import INR
from src.data.augmentations import setup_transformations


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
            window_size=config.model.window_size,
            num_heads=config.model.num_heads,
            input_dim=config.model.input_dim,
            latent_dim=config.model.latent_dim,
        )

        self.decoder = Decoder(
            max_positions=config.model.max_positions,
            num_layers=config.model.num_layers,
            d_model=config.model.d_model,
            dropout=config.model.dropout,
            window_size=config.model.window_size,
            num_heads=config.model.num_heads,
            input_dim=config.model.input_dim,
            latent_dim=config.model.latent_dim,
        )

        # Initialize projection head
        self.projection_head = ProjectionHead(
            d_model=config.model.latent_dim,
            n_tokens=config.model.n_tokens,
            output_dim=config.model.projection_dim,
        )

        # Initialize loss function
        self.loss_func = GammaContrastReconLoss(
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
        self.fixed_val_samples: list[Tensor] | None = None
        self.fixed_train_samples: list[Tensor] | None = None

        # Initialize weights
        self._initialize_weights()

        # Initialize transformations
        self.train_transforms, self.val_transforms, self.test_transforms = (
            setup_transformations(config)
        )

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

    @torch.no_grad()
    def visualize_reconstructions(self, samples, prefix: str, batch_idx: int):
        """Visualize reconstructions and log them."""

        if samples is None:
            print("Fixed samples not initialized.")
            return

        # Process batch
        tokens, masks, positions = samples

        latent, reconstructed, _ = self.forward(tokens, positions)

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

    # FIXME adjust
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

        loss_conf = self.loss_func(
            z_i=projected_i, z_j=projected_j, y=reconstructions, t=tokens, m=masks
        )

        loss_conv_conf = {
            f"{prefix}/loss_contrast": loss_conf["loss/loss_contrast"],
            f"{prefix}/loss_recon": loss_conf["loss/loss_recon"],
            f"{prefix}/loss": loss_conf["loss/loss"],
        }

        return loss_conv_conf[f"{prefix}/loss"], loss_conv_conf

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        original_tokens, original_masks, original_positions = batch
        tokens_i, masks_i, positions_i, tokens_j, masks_j, positions_j = (
            self.train_transforms(original_tokens, original_masks, original_positions)
        )
        batch_size = original_tokens.shape[0]

        # FORWARD PASS
        # I am combining the two views into one batch to make it faster
        combined_tokens = torch.cat([tokens_i, tokens_j], dim=0)
        combined_positions = torch.cat([positions_i, positions_j], dim=0)
        output1, output2, output3 = self.forward(combined_tokens, combined_positions)
        transform_output_i = (
            output1[:batch_size],
            output2[:batch_size],
            output3[:batch_size],
        )
        transform_output_j = (
            output1[batch_size:],
            output2[batch_size:],
            output3[batch_size:],
        )

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

        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        if (
            batch_idx == 0
            and self.current_epoch % self.config.logging.sample_every_n_epochs == 0
        ):
            # Log gradient norm
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

    # FIXME adjust
    def validation_step(self, batch, batch_idx: int) -> dict[str, Tensor]:
        original_tokens, original_masks, original_positions = batch
        tokens_i, masks_i, positions_i, tokens_j, masks_j, positions_j = (
            self.val_transforms(original_tokens, original_masks, original_positions)
        )
        batch_size = original_tokens.shape[0]

        # FORWARD PASS
        # I am combining the two views into one batch to make it faster
        combined_tokens = torch.cat([tokens_i, tokens_j], dim=0)
        combined_positions = torch.cat([positions_i, positions_j], dim=0)
        output1, output2, output3 = self.forward(combined_tokens, combined_positions)
        transform_output_i = (
            output1[:batch_size],
            output2[:batch_size],
            output3[:batch_size],
        )
        transform_output_j = (
            output1[batch_size:],
            output2[batch_size:],
            output3[batch_size:],
        )

        # Compute loss
        val_loss, val_log_dict = self.compute_loss(
            transform_output_i,
            transform_output_j,
            tokens_i,
            tokens_j,
            masks_i,
            masks_j,
            prefix="val",
        )

        # Log validation metrics
        self.log_dict(val_log_dict, prog_bar=True, sync_dist=True)

        if (
            batch_idx == 0
            and self.current_epoch % self.config.logging.sample_every_n_epochs == 0
        ):
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
