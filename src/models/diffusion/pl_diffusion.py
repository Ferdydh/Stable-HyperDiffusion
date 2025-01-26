from dataclasses import asdict
import numpy as np
import pytorch_lightning as pl
import torch
from einops import repeat
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import get_linear_schedule_with_warmup

from src.data.utils import generate_images
from src.models.diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from src.core.config_diffusion import DiffusionExperimentConfig
from src.models.utils import duplicate_batch_to_size, tokens_to_image_dict
from src.data.inr import INR
from src.models.diffusion.transformer import Transformer


def initialize_transformer(config: DiffusionExperimentConfig) -> Transformer:
    mlp = INR(up_scale=16)
    state_dict = mlp.state_dict()
    layers = []
    layer_names = []
    for layer in state_dict:
        shape = state_dict[layer].shape
        layers.append(np.prod(shape))
        layer_names.append(layer)
    #print(layers)
    #print(layer_names)

    return Transformer(
        layers,
        layer_names,
        **(asdict(config.transformer_config)),
    )


class HyperDiffusion(pl.LightningModule):
    def __init__(
        self,
        config: DiffusionExperimentConfig,
        image_shape: tuple,
        autoencoder: pl.LightningModule = None,
        positions:  torch.Tensor = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['autoencoder'])

        self.config = config
        self.image_shape = image_shape
        self.autoencoder = autoencoder
        
        if self.autoencoder is None:
            self.model = initialize_transformer(config)
        else:
            self.model = Transformer(
                                [520],
                                ['layer0'],
                                **(asdict(config.transformer_config)),
                            )
        self.positions = positions

        # Initialize diffusion model
        betas = torch.tensor(np.linspace(1e-4, 2e-2, config.timesteps))
        self.diff = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType[config.diff_config.model_mean_type],
            model_var_type=ModelVarType[config.diff_config.model_var_type],
            loss_type=LossType[config.diff_config.loss_type],
            diff_pl_module=self,
        )

        self.demo_inr = INR(up_scale=16)
        self.demo_inr.eval()

        # Initialize FID metric
        # self.fid = FrechetInceptionDistance(input_img_size=(3, 28, 28))

    def on_train_start(self):
        # Sanity check visualization
        tokens,_,positions = next(iter(self.trainer.train_dataloader))
        vis_dict = tokens_to_image_dict(
                tokens[0].unsqueeze(0), 
                positions[0].unsqueeze(0),
                self.demo_inr, 
                "train/original", 
                self.device,
                self.trainer.train_dataloader.dataset.get_state_dict(0)
            )
        self.logger.experiment.log(vis_dict)

        # Move autoencoder to device
        if self.autoencoder is not None:
            self.autoencoder = self.autoencoder.to(self.device)
            self.autoencoder.eval()
            tokens = tokens.to(self.device)
            positions = positions.to(self.device)
            with torch.no_grad():
                reconstructed, _,_,_ = self.autoencoder(tokens[0].unsqueeze(0), positions[0].unsqueeze(0))
            vis_dict = tokens_to_image_dict(
                reconstructed,  # tokens
                positions,  # pos
                self.demo_inr,
                "train/vae_reconstruction",
                self.device,
                self.trainer.train_dataloader.dataset.get_state_dict(0)
            )

            self.logger.experiment.log(vis_dict)

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

    def _compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        t = torch.randint(
            0, self.diff.num_timesteps, (batch.shape[0],), device=self.device
        )
        loss_terms = self.diff.training_losses(
            self.model,
            batch,
            t,
        )
        return loss_terms["loss"].mean()

    def training_step(self, batch, batch_idx):
        original_tokens, _, original_positions = batch
        original_tokens = original_tokens.to(self.device)
        original_positions = original_positions.to(self.device)
        
        with torch.no_grad():
            latent_vector,_,_ = self.autoencoder.encoder(original_tokens, original_positions)    # Shape: (batch_size, n_tokens, latent_dim)

        flattened_latent = latent_vector.view(latent_vector.shape[0], -1)

        batch = duplicate_batch_to_size(flattened_latent)
        loss = self._compute_loss(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        original_tokens, _, original_positions = batch
        original_tokens = original_tokens.to(self.device)
        original_positions = original_positions.to(self.device)
        
        with torch.no_grad():
            latent_vector,_,_ = self.autoencoder.encoder(original_tokens, original_positions)

        flattened_latent = latent_vector.view(latent_vector.shape[0], -1)
        #print(flattened_latent.shape)

        loss = self._compute_loss(flattened_latent)
        # More explicit logging
        self.log("val/loss", loss)

        return loss

    def on_train_epoch_end(self):
        if self.current_epoch % self.config.visualize_every_n_epochs == 0:
            
            samples_tokenized, samples_reconstructed, positions = self.generate_samples(num_samples=self.config.logging.num_samples_to_visualize)

            vis_dict = tokens_to_image_dict(
                samples_reconstructed, 
                positions,
                self.demo_inr, 
                "train/reconstruction", 
                self.device,
                self.trainer.train_dataloader.dataset.get_state_dict(0)
            )
            vis_dict["epoch"] = self.current_epoch
            self.logger.experiment.log(vis_dict)

            # Calculate FID if it's time
            # if self.current_epoch % self.config.val_fid_calculation_period == 0:
            #     # Generate more samples for FID
            #     samples = []
            #     num_samples = self.config.num_samples_metrics
            #     batch_size = min(100, num_samples)

            #     for idx in range(0, num_samples, batch_size):
            #         curr_batch_size = min(batch_size, num_samples - idx)
            #         sample = self.diff.ddim_sample_loop(
            #             self.model, (curr_batch_size, *self.image_shape[1:])
            #         )
            #         samples.append(sample)

            #     fake_images = generate_images(
            #         torch.vstack(samples), self.demo_inr, self.device
            #     )

            #     # Get training images for FID
            #     train_dataset = self.trainer.train_dataloader.dataset
            #     max_samples = min(len(train_dataset), num_samples)
            #     real_samples = torch.vstack(
            #         [train_dataset[i] for i in range(max_samples)]
            #     )
            #     real_images = generate_images(real_samples, self.demo_inr, self.device)

            #     # Compute and log FID
            #     # Expand single channel to RGB
            #     real_images = repeat(real_images, "b h w -> b c h w", c=3)
            #     fake_images = repeat(fake_images, "b h w -> b c h w", c=3)

            #     self.fid.update(real_images.cuda(), real=True)
            #     self.fid.update(fake_images.cuda(), real=False)
            #     fid_score = self.fid.compute()

            #     self.log("metrics/fid", fid_score)

    def generate_samples(self, num_samples=10):
        self.eval()
        # Generate samples for visualization
        samples = self.diff.ddim_sample_loop(self.model, (num_samples, *self.image_shape[1:]))    # Output: BATCH_SIZE x 520  -> as latent_dim * n_tokens = 8 * 65 = 520
        #print(samples.shape)
        samples_tokenized = samples.view(samples.shape[0], self.autoencoder.config.model.n_tokens, self.autoencoder.config.model.latent_dim)
        #print(samples_tokenized.shape)

        positions = self.positions.unsqueeze(0).repeat(samples_tokenized.shape[0], 1, 1).to(self.device)

        with torch.no_grad():
            samples_reconstructed = self.autoencoder.decoder(samples_tokenized, positions)
        #print(samples_reconstructed.shape)
        return samples_tokenized, samples_reconstructed, positions