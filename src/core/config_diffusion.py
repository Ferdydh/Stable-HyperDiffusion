from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.core.config import (
    BaseExperimentConfig,
    CheckpointConfig,
    DataConfig,
    EarlyStoppingConfig,
    LoggingConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    get_device,
)


@dataclass
class DiffusionConfig:
    """Configuration for diffusion model parameters"""

    model_mean_type: str
    model_var_type: str
    loss_type: str

    @classmethod
    def default(cls) -> "DiffusionConfig":
        return cls(
            model_mean_type="START_X", model_var_type="FIXED_LARGE", loss_type="MSE"
        )


@dataclass
class TransformerConfig:
    """Configuration for transformer architecture"""

    n_embd: int
    n_layer: int
    n_head: int
    split_policy: str
    use_global_residual: bool
    condition: str
    chunk_size: int | None = None

    @classmethod
    def default(cls) -> "TransformerConfig":
        return cls(
            n_embd=2880,
            n_layer=12,
            n_head=16,
            split_policy="layer_by_layer",
            use_global_residual=False,
            condition="no",
        )


@dataclass
class DiffusionExperimentConfig(BaseExperimentConfig):
    """Configuration for diffusion experiments"""

    num_samples_metrics: int
    test_sample_mult: float
    timesteps: int
    scheduler_step: int
    best_model_save_path: Optional[str]
    mode: str
    val_fid_calculation_period: int
    visualize_every_n_epochs: int
    lr: float
    accumulate_grad_batches: int
    diff_config: DiffusionConfig
    transformer_config: TransformerConfig
    autoencoder_checkpoint: Optional[str]

    @classmethod
    def sanity(cls) -> "DiffusionExperimentConfig":
        return cls(
            # Base config defaults
            data=DataConfig.sanity(),
            optimizer=OptimizerConfig.default(),
            scheduler=SchedulerConfig.cosine_default(),
            trainer=TrainerConfig.sanity(),
            logging=LoggingConfig.sanity(),
            checkpoint=CheckpointConfig.default(),
            early_stopping=EarlyStoppingConfig.default(),
            device=get_device(),
            #
            #
            # Diffusion-specific defaults
            test_sample_mult=1.1,
            timesteps=500,
            scheduler_step=200,
            best_model_save_path=None,
            mode="train",
            val_fid_calculation_period=15,
            visualize_every_n_epochs=100,
            lr=0.0002,
            accumulate_grad_batches=1,
            diff_config=DiffusionConfig.default(),
            transformer_config=TransformerConfig.default(),
            autoencoder_checkpoint=None,
            num_samples_metrics=4,
        )
