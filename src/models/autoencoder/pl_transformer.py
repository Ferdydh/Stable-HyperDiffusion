from dataclasses import asdict
import torch
import pytorch_lightning as pl
from torch import Tensor
from typing import Tuple, Dict, List, Optional
import torch.nn as nn
import math

from src.core.config import TransformerExperimentConfig
from src.models.utils import (
    weights_to_image_dict,
)
from data.data_converter import tokens_to_weights
from src.models.autoencoder.losses import GammaContrastReconLoss
from src.models.autoencoder.transformer import Encoder, Decoder, ProjectionHead
from src.data.inr import INR
from src.data.augmentations import (
    AugmentationPipeline,
    TwoViewSplit,
    WindowCutter,
    ErasingAugmentation,
    NoiseAugmentation,
    MultiWindowCutter,
    StackBatches,
    PermutationSelector,
)
# from src.models.utils import (
#     create_reconstruction_visualizations_with_state_dict as create_reconstruction_visualizations,
# )


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
        self.setup_transformations()

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

    def on_train_start(self):
        """Setup fixed validation and training samples for visualization."""
        num_samples = self.config.logging.num_samples_to_visualize

        if (
            len(self.trainer.val_dataloaders.dataset) < num_samples
            or len(self.trainer.train_dataloader.dataset) < num_samples
        ):
            raise ValueError(
                f"Number of samples to visualize ({num_samples}) is greater than the number of samples in the dataset ({len(self.trainer.val_dataloaders.dataset)})."
            )

        self.fixed_val_samples = [
            self.trainer.val_dataloaders.dataset.get_state_dict(i)
            for i in range(num_samples)
        ]

        self.fixed_train_samples = [
            self.trainer.train_dataloader.dataset.get_state_dict(i)
            for i in range(num_samples)
        ]

        v = weights_to_image_dict(
            self.fixed_val_samples, self.demo_inr, "val/original", self.device
        )
        t = weights_to_image_dict(
            self.fixed_train_samples, self.demo_inr, "train/original", self.device
        )

        self.logger.experiment.log(v)
        self.logger.experiment.log(t)

    def setup_transformations(self):
        # set windowsize
        windowsize = self.config.model.window_size
        # TRAIN AUGMENTATIONS
        stack_1 = []
        # if config.get("trainset::add_noise_view_1", 0.0) > 0.0:
        #    stack_1.append(NoiseAugmentation(config.get("trainset::add_noise_view_1", 0.0)))
        if self.config.augmentations.add_noise_view_1_train > 0.0:
            stack_1.append(
                NoiseAugmentation(self.config.augmentations.add_noise_view_1_train)
            )
        # if config.get("trainset::erase_augment", None) is not None:
        #    stack_1.append(ErasingAugmentation(**config["trainset::erase_augment"]))
        if self.config.augmentations.erase_augment_view_1_train is not None:
            stack_1.append(
                ErasingAugmentation(
                    self.config.augmentations.erase_augment_view_1_train
                )
            )
        stack_2 = []
        # if config.get("trainset::add_noise_view_2", 0.0) > 0.0:
        #    stack_2.append(NoiseAugmentation(config.get("trainset::add_noise_view_2", 0.0)))
        if self.config.augmentations.add_noise_view_2_train > 0.0:
            stack_2.append(
                NoiseAugmentation(self.config.augmentations.add_noise_view_2_train)
            )
        # if config.get("trainset::erase_augment", None) is not None:
        #    stack_2.append(ErasingAugmentation(**config["trainset::erase_augment"]))
        if self.config.augmentations.erase_augment_view_2_train is not None:
            stack_2.append(
                ErasingAugmentation(
                    self.config.augmentations.erase_augment_view_2_train
                )
            )

        stack_train = []
        # if config.get("trainset::multi_windows", None):
        #   stack_train.append(StackBatches())
        if self.config.augmentations.multi_windows_train:
            stack_train.append(StackBatches())
        else:
            stack_train.append(WindowCutter(windowsize=windowsize))
        # put train stack together
        # if config.get("training::permutation_number", 0) == 0:
        if self.config.augmentations.permutation_number_train == 0:
            split_mode = "copy"
            view_1_canon = True
            view_2_canon = True
        else:
            split_mode = "permutation"
            # view_1_canon = config.get("training::view_1_canon", True)
            view_1_canon = self.config.augmentations.view_1_canon_train
            # view_2_canon = config.get("training::view_2_canon", False)
            view_2_canon = self.config.augmentations.view_2_canon_train
        stack_train.append(
            TwoViewSplit(
                stack_1=stack_1,
                stack_2=stack_2,
                mode=split_mode,
                view_1_canon=view_1_canon,
                view_2_canon=view_2_canon,
            ),
        )

        trafo_train = AugmentationPipeline(stack=stack_train)

        # test AUGMENTATIONS
        stack_1 = []
        # if config.get("testset::add_noise_view_1", 0.0) > 0.0:
        #    stack_1.append(NoiseAugmentation(config.get("testset::add_noise_view_1", 0.0)))
        if self.config.augmentations.add_noise_view_1_val > 0.0:
            stack_1.append(
                NoiseAugmentation(self.config.augmentations.add_noise_view_1_val)
            )
        # if config.get("testset::erase_augment", None) is not None:
        #    stack_1.append(ErasingAugmentation(**config["testset::erase_augment"]))
        if self.config.augmentations.erase_augment_view_1_val is not None:
            stack_1.append(
                ErasingAugmentation(self.config.augmentations.erase_augment_view_1_val)
            )
        stack_2 = []
        # if config.get("testset::add_noise_view_2", 0.0) > 0.0:
        #    stack_2.append(NoiseAugmentation(config.get("testset::add_noise_view_2", 0.0)))
        if self.config.augmentations.add_noise_view_2_val > 0.0:
            stack_2.append(
                NoiseAugmentation(self.config.augmentations.add_noise_view_2_val)
            )
        # if config.get("testset::erase_augment", None) is not None:
        #    stack_2.append(ErasingAugmentation(**config["testset::erase_augment"]))
        if self.config.augmentations.erase_augment_view_2_val is not None:
            stack_2.append(
                ErasingAugmentation(self.config.augmentations.erase_augment_view_2_val)
            )

        stack_test = []
        # if config.get("trainset::multi_windows", None):
        if self.config.augmentations.multi_windows_train:
            stack_test.append(StackBatches())
        else:
            stack_test.append(WindowCutter(windowsize=windowsize))
        # put together
        # if config.get("testing::permutation_number", 0) == 0:
        if self.config.augmentations.permutation_number_val == 0:
            split_mode = "copy"
            view_1_canon = True
            view_2_canon = True
        else:
            split_mode = "permutation"
            # view_1_canon = config.get("testing::view_1_canon", True)
            view_1_canon = self.config.augmentations.view_1_canon_val
            # view_2_canon = config.get("testing::view_2_canon", False)
            view_2_canon = self.config.augmentations.view_2_canon_val
        stack_test.append(
            TwoViewSplit(
                stack_1=stack_1,
                stack_2=stack_2,
                mode=split_mode,
                view_1_canon=view_1_canon,
                view_2_canon=view_2_canon,
            ),
        )

        # TODO: pass through permutation / view_1/2 canonical
        trafo_test = AugmentationPipeline(stack=stack_test)

        # downstream task permutation (chose which permutationn to use for dstk)
        # if config.get("training::permutation_number", 0) > 0:
        if self.config.augmentations.permutation_number_train > 0:
            trafo_dst = PermutationSelector(mode="canonical", keep_properties=True)
        else:
            trafo_dst = PermutationSelector(mode="identity", keep_properties=True)

        self.train_transforms = trafo_train
        self.val_transforms = trafo_test
        self.dst_transform = trafo_dst

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

    # FIXME adjust
    def visualize_reconstructions(self, prefix: str, batch_idx: int):
        """Visualize reconstructions and log them."""
        if (
            batch_idx % self.config.logging.log_every_n_steps != 0
            or self.current_epoch == 0
        ):
            return

        with torch.no_grad():
            num_samples = self.config.logging.num_samples_to_visualize
            if prefix == "train":
                samples = [
                    self.trainer.train_dataloader.dataset[i] for i in range(num_samples)
                ]
            else:
                samples = [
                    self.trainer.val_dataloaders.dataset[i] for i in range(num_samples)
                ]

            # Process batch
            tokens, masks, positions = zip(*samples)
            tokens, masks, positions = (
                torch.stack(tokens).to(self.device),
                torch.stack(masks).to(self.device),
                torch.stack(positions).to(self.device).to(torch.int32),
            )

            latent, reconstructed, _ = self.forward(tokens, positions)

            reference_checkpoint = self.trainer.val_dataloaders.dataset.get_state_dict(
                0
            )

            reconstructions = [
                tokens_to_weights(t, p, reference_checkpoint)
                for t, p in zip(reconstructed, positions)
            ]

            # step_key = f"{prefix}_step_{self.global_step}"
            # self.fixed_sample_reconstructions[step_key] = reconstructions

            vis_dict = weights_to_image_dict(
                # samples,
                reconstructions,
                self.demo_inr,
                f"{prefix}/reconstruction",
                # batch_idx,
                # self.global_step,
                # is_fixed=True,
                self.device,
            )
            vis_dict["global_step"] = self.global_step
            self.logger.experiment.log(vis_dict)

    # FIXME No longer needed
    # def _process_batch(self, batch: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    #    """Convert batch to required tensor format."""
    #    tokens, masks, positions = zip(
    #        *[weights_to_tokens(b, tokensize=0, device=self.device) for b in batch]
    #    )
    #    return (
    #        torch.stack(tokens).to(self.device),
    #        torch.stack(masks).to(self.device),
    #        torch.stack(positions).to(self.device).to(torch.int32),
    #    )

    # FIXME adjust
    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """Execute a single training step."""

        # Process batch
        tokens, masks, positions = zip(*batch)
        tokens, masks, positions = (
            torch.stack(tokens),
            torch.stack(masks),
            torch.stack(positions).to(torch.int32),
        )

        # Get transformed versions
        transformed = self.train_transforms(tokens, masks, positions)

        tokens_i, masks_i, positions_i = tokens, masks, positions
        tokens_j, masks_j, positions_j = transformed[0], transformed[1], transformed[2]

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

        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        if batch_idx % self.config.trainer.log_every_n_steps == 0:
            # self.visualize_reconstructions(
            #    batch, "train", batch_idx
            # )
            self.visualize_reconstructions("train", batch_idx)

        # Log gradient norm
        # if batch_idx % self.config.trainer.log_every_n_steps == 0:
        #     total_norm = 0.0
        #     for p in self.parameters():
        #         if p.grad is not None:
        #             param_norm = p.grad.data.norm(2)
        #             total_norm += param_norm.item() ** 2
        #     total_norm = total_norm**0.5
        #     self.log("train/grad_norm", total_norm, prog_bar=False, sync_dist=True)

        #     # Visualize both fixed samples and current batch
        #     self.visualize_reconstructions(self.fixed_train_samples, "train", batch_idx)

        return loss

    # FIXME adjust
    def validation_step(self, batch, batch_idx: int) -> dict[str, Tensor]:
        # Process batch
        tokens, masks, positions = zip(*batch)
        tokens, masks, positions = (
            torch.stack(tokens),
            torch.stack(masks),
            torch.stack(positions).to(torch.int32),
        )

        # Get transformed versions
        transformed = self.val_transforms(*[tokens, masks, positions])
        # print("Tokens:",tokens.shape)
        # print("Masks:",masks.shape)
        # print("Positions:",positions.shape)
        # for x in transformed:
        #    print(x.shape)
        tokens_i, masks_i, positions_i = transformed[0], transformed[1], transformed[2]
        tokens_j, masks_j, positions_j = transformed[3], transformed[4], transformed[5]
        if tokens_i.ndim == 2:
            tokens_i = tokens_i.unsqueeze(0)
            tokens_j = tokens_j.unsqueeze(0)

        # Forward pass
        transform_output_i = self.forward(tokens_i, positions_i)
        transform_output_j = self.forward(tokens_j, positions_j)

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
            self.visualize_reconstructions("val", batch_idx)

        return val_log_dict

        """
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
        """

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
