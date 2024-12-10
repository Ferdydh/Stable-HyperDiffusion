import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
import wandb
from datetime import datetime
import atexit

from core.utils import load_config, get_device
from models.diffusion.diffusion import INRDiffusion
from data.inr_dataset import (
    DataHandler,
    create_selector_from_config,
)


def cleanup_wandb():
    """Ensure wandb run is properly closed."""
    try:
        wandb.finish()
    except:
        pass


def train_diffusion(experiment: str = "diffusion_sanity_check"):
    """Train the diffusion model with improved logging and visualization."""
    # Register cleanup function
    atexit.register(cleanup_wandb)

    try:
        # Load configuration
        cfg = load_config(experiment)
        device = get_device()

        # Setup unique run name if not specified
        if cfg["logging"]["run_name"] is None:
            cfg["logging"]["run_name"] = (
                f"diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        # Initialize WandB logger with modified settings
        wandb_logger = WandbLogger(
            project=cfg["logging"]["project_name"],
            name=cfg["logging"]["run_name"],
            log_model=cfg["logging"]["log_model"],
            save_dir=cfg["logging"]["save_dir"],
            settings=wandb.Settings(start_method="thread"),
        )

        # Log hyperparameters and config file
        wandb_logger.log_hyperparams(cfg)
        wandb.save(experiment)

        # Initialize data handler
        data_handler = DataHandler(
            hparams={**cfg["data"], "device": device},
            data_folder=cfg["data"]["data_path"],
            selectors=create_selector_from_config(cfg),
        )

        # Get dataloaders
        train_loader = data_handler.train_dataloader()
        val_loader = data_handler.val_dataloader()

        # Setup callbacks
        callbacks = []

        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg["checkpoint"]["dirpath"],
            filename=cfg["checkpoint"]["filename"],
            monitor=cfg["checkpoint"]["monitor"],
            mode=cfg["checkpoint"]["mode"],
            save_last=cfg["checkpoint"]["save_last"],
            save_top_k=cfg["checkpoint"]["save_top_k"],
        )
        callbacks.append(checkpoint_callback)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor=cfg["early_stopping"]["monitor"],
            min_delta=cfg["early_stopping"]["min_delta"],
            patience=cfg["early_stopping"]["patience"],
            verbose=True,
            mode=cfg["early_stopping"]["mode"],
            check_finite=True,
            stopping_threshold=cfg["early_stopping"].get("stopping_threshold", None),
            divergence_threshold=cfg["early_stopping"].get(
                "divergence_threshold", None
            ),
        )
        callbacks.append(early_stopping)

        # Initialize trainer with diffusion-specific settings
        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=callbacks,
            max_epochs=cfg["training"][
                "epochs"
            ],  # Note: using training config instead of trainer
            accelerator="auto",
            devices="auto",
            precision=32,  # Diffusion typically needs full precision
            gradient_clip_val=cfg["training"]["gradient_clip_val"],
            accumulate_grad_batches=cfg["training"]["accumulate_grad_batches"],
            log_every_n_steps=cfg["logging"]["log_every_n_steps"],
            val_check_interval=0.5,  # Check validation every half epoch
        )

        # Add this right before trainer.fit()
        wandb.require("service")

        # Initialize diffusion model
        model = INRDiffusion(cfg)

        # Train model
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        raise

    finally:
        # Ensure wandb is properly closed
        cleanup_wandb()
        print("\nWandB run closed. Exiting...")
