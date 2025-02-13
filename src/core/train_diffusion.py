from dataclasses import asdict
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
import os
import torch

import wandb
from datetime import datetime
import atexit

from src.core.config import TransformerExperimentConfig
from src.core.config_diffusion import DiffusionExperimentConfig
from src.data.inr_dataset import DataHandler
from src.models.autoencoder.pl_transformer import Autoencoder
from src.models.diffusion.pl_diffusion import HyperDiffusion
from src.data.data_converter import weights_to_flattened_weights

torch.set_float32_matmul_precision("high")


def cleanup_wandb():
    """Ensure wandb run is properly closed."""
    try:
        wandb.finish()
    except:
        pass


def train(
    config: DiffusionExperimentConfig,
    config_ae: TransformerExperimentConfig = None,
    continue_training_from_path: str = None,
):
    """Train the autoencoder model with improved logging and visualization."""
    # Register cleanup function
    atexit.register(cleanup_wandb)

    try:
        # Load configuration
        # Setup unique run name if not specified
        if config.logging.run_name is None:
            config.logging.run_name = (
                f"autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            config.logging.run_name = (
                f"{config.logging.run_name} {datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        autoencoder = None

        # Training on whole stable hyperdiffusion model (including pretrained autoencoder)
        use_autoencoder = (
            config.autoencoder_checkpoint
            and config_ae is not None
            and config.autoencoder_checkpoint is not None
        )

        if use_autoencoder:
            # TODO: add config
            autoencoder = Autoencoder(config=config_ae)
            state_dict = torch.load(config.autoencoder_checkpoint)
            autoencoder.load_state_dict(state_dict["state_dict"])

            for param in autoencoder.parameters():
                param.requires_grad = False
            autoencoder.eval()

        # Initialize data handler
        data_handler = DataHandler(config, use_autoencoder)

        data_handler.setup()

        # Initialize WandB logger with modified settings
        wandb_logger = WandbLogger(
            project=config.logging.project_name,
            name=config.logging.run_name,
            log_model=config.logging.log_model,
            save_dir=config.logging.save_dir,
            settings=wandb.Settings(start_method="thread"),
            dir="diffusion_logs",
        )

        # Log hyperparameters and config file
        wandb_logger.log_hyperparams(asdict(config))

        positions = None

        # Training on whole stable hyperdiffusion model (including pretrained autoencoder)
        if config_ae:
            data_shape = (
                config.data.batch_size,
                config_ae.model.n_tokens * config_ae.model.latent_dim,
            )
            print(data_shape)
            _, _, positions = data_handler.train_dataloader().dataset[0]
        else:
            data_shape = next(iter(data_handler.train_dataloader())).shape

        # Initialize model
        diffuser_opt = HyperDiffusion(config, data_shape, autoencoder, positions)

        if continue_training_from_path is not None:
            state_dict = torch.load(continue_training_from_path)
            diffuser_opt.load_state_dict(state_dict["state_dict"])

        # diffuser_opt = torch.compile(diffuser)

        # checkpoint = torch.load("logs\\hyperdiffusion\\1ve81sk7\\checkpoints\\epoch=929-step=7440.ckpt")
        # diffuser.load_state_dict(checkpoint["state_dict"])

        # diffuser = HyperDiffusion.load_from_checkpoint("logs\hyperdiffusion\2pc1g4zc\checkpoints\epoch=434-step=85695.ckpt")

        # Specify where to save checkpoints
        checkpoint_path = os.path.join(
            "diffusion_logs",
            "lightning_checkpoints",
            f"{str(datetime.now()).replace(':', '-') + '-' + wandb.run.name + '-' + wandb.run.id}",
        )

        # Setup callbacks
        callbacks = []
        # last_model_saver = ModelCheckpoint(
        #     dirpath=checkpoint_path,
        #     filename="last-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
        #     save_on_train_epoch_end=True,
        # )
        # callbacks.append(last_model_saver)
        # best_fid_checkpoint = ModelCheckpoint(
        #     save_top_k=1,
        #     monitor="val/fid",
        #     mode="min",
        #     dirpath=checkpoint_path,
        #     filename="best-val-fid-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
        # )
        # callbacks.append(best_fid_checkpoint)

        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_path,
            filename="best-loss-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
            monitor="val/loss",
            # monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_last=config.checkpoint.save_last,
            save_top_k=config.checkpoint.save_top_k,
        )
        callbacks.append(checkpoint_callback)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # Early stopping callback
        # early_stopping = EarlyStopping(
        #     monitor=config.early_stopping.monitor,
        #     min_delta=config.early_stopping.min_delta,
        #     patience=config.early_stopping.patience,
        #     verbose=True,
        #     mode=config.early_stopping.mode,
        #     check_on_train_epoch_end=False,
        #     check_finite=True,  # Stop if loss becomes NaN or inf
        # )
        # callbacks.append(early_stopping)

        # Initialize trainer
        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=callbacks,
            max_epochs=config.trainer.max_epochs,
            accelerator="auto",
            devices="auto",
            precision=config.trainer.precision,
            gradient_clip_val=config.trainer.gradient_clip_val,
            log_every_n_steps=config.trainer.log_every_n_steps,
            check_val_every_n_epoch=1,
            # num_sanity_val_steps=0,
            accumulate_grad_batches=config.accumulate_grad_batches,
        )

        # Add this right before trainer.fit()
        wandb.require("service")

        trainer.fit(model=diffuser_opt, datamodule=data_handler)
        # best_model_save_path is the path to saved best model
        # trainer.test(
        #    diffuser,
        #    test_dl,
        #    ckpt_path=best_model_save_path if Config.get("mode") == "test" else None,
        # )
        # wandb_logger.finalize("Success")
        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        raise

    finally:
        # Ensure wandb is properly closed
        cleanup_wandb()
        print("\nWandB run closed. Exiting...")
