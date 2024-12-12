import torch
import pytorch_lightning as pl
from torch import Tensor
from typing import Tuple
from jaxtyping import Float
from typeguard import typechecked
import torch.nn as nn

from data.utils import weights_to_tokens
from models.autoencoder.losses import GammaContrastReconLoss
from models.autoencoder.transformer import Transformer
from models.inr import INR
from models.utils import create_reconstruction_visualizations


class SimpleProjectionHead(nn.Module):
    """
    Projection head: maps sequences of token embeddings and maps them to embeddings
    """

    def __init__(self, d_model: int = 512, n_tokens: int = 12, odim: int = 50):
        super(SimpleProjectionHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(d_model * n_tokens, odim, bias=False),
            nn.LayerNorm(odim),
            nn.ReLU(),
            nn.Linear(odim, odim, bias=False),
            nn.LayerNorm(odim),
            nn.ReLU(),
        )

    def forward(self, z: torch.tensor) -> torch.tensor:
        """
        Args:
            z: sequence of token embeddings [nbatch,token_window,token_dim]
        """
        # avereage tokens
        # z = z.mean(dim=1)
        z = z.view(z.shape[0], -1)
        # pass through head
        z = self.head(z)
        # return
        return z


class PositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs.
    Attributes:
        posemb_init: positional embedding initializer.
        max_positions: maximum number of positions to embed.
        embedding_dim: dimension of the input embeddings.
    """

    def __init__(self, max_positions=[48, 256], embedding_dim=128):
        super().__init__()
        self.max_positions = max_positions
        self.embedding_dim = embedding_dim
        if len(max_positions) == 2:
            self.pe1 = nn.Embedding(max_positions[0], embedding_dim // 2)
            self.pe2 = nn.Embedding(max_positions[1], embedding_dim // 2)
            self.pe3 = None
        elif len(max_positions) == 3:
            self.pe1 = nn.Embedding(max_positions[0], embedding_dim // 2)  # add 1 + 2
            self.pe2 = nn.Embedding(max_positions[1], embedding_dim // 2)  # add 1 + 2
            self.pe3 = nn.Embedding(max_positions[2], embedding_dim // 2)  # cat 1+2 & 3

    def forward(self, inputs, pos):
        """Applies the AddPositionEmbs module.
        Args:
            inputs: Inputs to the layer, shape `(batch_size, seq_len, emb_dim)`.
            pos: Position of the first token in each sequence, shape `(batch_size,seq_len,2)`.
        Returns:
            Output tensor with shape `(batch_size, seq_len, emb_dim + 2)`.
        """
        assert (
            inputs.ndim == 3
        ), f"Number of dimensions should be 3, but it is {inputs.ndim}"
        assert pos.shape[2] == len(
            self.max_positions
        ), "Position tensors should have as many demsions as max_positions"
        assert (
            pos.shape[0] == inputs.shape[0]
        ), "Position tensors should have the same batch size as inputs"
        assert (
            pos.shape[1] == inputs.shape[1]
        ), "Position tensors should have the same seq length as inputs"
        pos = pos.int()
        pos_emb1 = self.pe1(pos[:, :, 0])
        pos_emb2 = self.pe2(pos[:, :, 1])
        if self.pe3 is not None:
            pos_emb3 = self.pe3(pos[:, :, 2])
            pos_emb = [pos_emb1 + pos_emb2, pos_emb3]
        else:
            pos_emb = [pos_emb1, pos_emb2]

        pos_emb = torch.cat(pos_emb, dim=2)

        out = inputs + pos_emb
        return out


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

        # TODO!!!
        max_positions = [100, 10, 40]
        num_layers = 8
        d_model = 1024
        dropout = 0.0
        windowsize = 32
        nhead = 8
        i_dim = 33
        lat_dim = 128

        self.tokenizer = nn.Linear(i_dim, d_model)

        self.pe = PositionEmbs(max_positions=max_positions, embedding_dim=d_model)

        # TODO set this
        self.transformer = Transformer(
            n_layer=num_layers,
            n_head=nhead,
            d_model=d_model,
            dropout=dropout,
            bias=False,
            causal=False,
            block_size=windowsize,
        )
        self.encoder_comp = nn.Linear(d_model, lat_dim)

    @typechecked
    def forward(
        self,
        x,
        p,
        mask,
    ) -> Tensor:
        # map weight tokens from input dim to d_model
        x = self.tokenizer(x)
        # add position embeddings
        x = self.pe(x, p)
        # pass through encoder transformer
        x = self.transformer(x, mask=mask)
        # compress to latent dim
        z = self.encoder_comp(x)

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

        # TODO: put in dictionary
        max_positions = [100, 10, 40]
        d_model = 1024
        num_layers = 8
        nhead = 8
        windowsize = 32
        dropout = 0.0
        lat_dim = 128
        i_dim = 33

        self.decoder_comp = nn.Linear(lat_dim, d_model)
        self.pe = PositionEmbs(max_positions=max_positions, embedding_dim=d_model)
        self.transformer = Transformer(
            n_layer=num_layers,
            n_head=nhead,
            d_model=d_model,
            dropout=dropout,
            bias=False,
            causal=False,
            block_size=windowsize,
        )
        self.detokenizer = nn.Linear(d_model, i_dim)

    @typechecked
    def forward(
        self, z, p, mask
    ) -> Tensor:
        x = self.decoder_comp(z)
        x = self.pe(x, p)
        x = self.transformer(x, mask=mask)
        x_reconstructed = self.detokenizer(x)
        return x_reconstructed


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

        # Initialize quality metrics
        self.best_val_loss = float("inf")

        # Create demo INR for visualization
        self.demo_inr = INR(up_scale=16)
        # Move demo INR to the same device as the model
        self.demo_inr = self.demo_inr.to(self.device)

        # TODO: put in dictionary
        lat_dim = 128
        windowsize = 32
        odim=30

        self.projection_head = SimpleProjectionHead(
            d_model=lat_dim, n_tokens=windowsize, odim=odim
        )

        self.criterion = GammaContrastReconLoss(
            gamma=config.get("training::gamma", 0.5),
            reduction=config.get("training::reduction", "mean"),
            batch_size=config.get("trainset::batchsize", 64),
            temperature=config.get("training::temperature", 0.1),
            contrast=config.get("training::contrast", "simclr"),
            z_var_penalty=config.get("training::z_var_penalty", 0.0),
        )

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

    @typechecked
    def encode(
        self, x, p: torch.tensor, mask=None
    ) -> Tensor:
        return self.encoder(x, p, mask)

    @typechecked
    def decode(
        self, z: Tensor, p: Tensor, mask=None
    ) -> Tensor:
        return self.decoder(
            z,
            p,
            mask,
        )

    @typechecked
    def forward(
        self, input, p, mask
    ) -> Tuple[Tensor, Tensor, Tensor]:
        z = self.encode(input, p, mask)
        zp = self.projection_head(z)
        dec = self.decode(z)
        return z, dec, zp

    def compute_loss(
        self,
        inputs,
        reconstructions,
        masks,
        positions,
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


    def training_step(self, batch, batch_idx: int) -> Tensor:
        # Convert batch to tensors all at once instead of loop
        tokens, masks, positions = zip(*[weights_to_tokens(b, tokensize=0, device=self.device) for b in batch])

        # Stack the tensors along batch dimension
        tokens = torch.stack(tokens).to(self.device)
        masks = torch.stack(masks).to(self.device)
        positions = torch.stack(positions).to(self.device)

        print(f"tokens shape: {tokens.shape}")
        print(f"masks shape: {masks.shape}")
        print(f"positions shape: {positions.shape}")

        # Forward pass
        z, reconstructions, zp = self.forward(tokens, positions, masks)

        # TODO: continue

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

    def validation_step(self, batch, batch_idx: int) -> dict[str, Tensor]:
        # Convert batch to tensors all at once instead of loop
        tokens, masks, positions = zip(*[weights_to_tokens(b, tokensize=0, device=self.device) for b in batch])

        # Stack the tensors along batch dimension
        tokens = torch.stack(tokens).to(self.device)
        masks = torch.stack(masks).to(self.device)
        positions = torch.stack(positions).to(self.device)

        print(f"tokens shape: {tokens.shape}")
        print(f"masks shape: {masks.shape}")
        print(f"positions shape: {positions.shape}")
        print(tokens)

        # Forward pass
        z, reconstructions, zp = self.forward(tokens, positions, masks)
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
