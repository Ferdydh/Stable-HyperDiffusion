from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.config import (
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
class MLPConfig:
    """Configuration for MLP architecture"""

    model_type: str
    out_size: int
    hidden_neurons: List[int]
    output_type: str
    out_act: str
    multires: int
    use_leaky_relu: bool
    move: bool

    @classmethod
    def default(cls) -> "MLPConfig":
        return cls(
            model_type="mlp_3d",
            out_size=1,
            hidden_neurons=[128, 128, 128],
            output_type="occ",
            out_act="sigmoid",
            multires=4,
            use_leaky_relu=False,
            move=False,
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

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_embd": self.n_embd,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "split_policy": self.split_policy,
            "use_global_residual": self.use_global_residual,
            "condition": self.condition,
        }


@dataclass
class ValidationConfig:
    """Configuration for validation parameters"""

    num_points: int
    num_samples_metrics: int
    num_samples_visualization: int
    visualize_every_n_epochs: int

    @classmethod
    def default(cls) -> "ValidationConfig":
        return cls(
            num_points=2048,
            num_samples_metrics=60,
            num_samples_visualization=4,
            visualize_every_n_epochs=20,
        )


@dataclass
class DiffusionExperimentConfig(BaseExperimentConfig):
    """Configuration for diffusion experiments"""

    method: str
    calculate_metric_on_test: bool
    dedup: bool
    test_sample_mult: float
    filter_bad: bool
    filter_bad_path: str
    disable_wandb: bool
    normalization_factor: int
    timesteps: int
    use_scheduler: bool
    scheduler_step: int
    best_model_save_path: Optional[str]
    mode: str
    model_resume_path: Optional[str]
    sampling: str
    val_fid_calculation_period: int
    visualize_every_n_epochs: int
    lr: float
    accumulate_grad_batches: int
    val: ValidationConfig
    mlp_config: MLPConfig
    diff_config: DiffusionConfig
    transformer_config: TransformerConfig
    autoencoder_checkpoint: Optional[str]

    @classmethod
    def sanity(cls) -> "DiffusionExperimentConfig":
        return cls(
            # Base config defaults
            # data=DataConfig.sanity(),
            # optimizer=OptimizerConfig.default(),
            # scheduler=SchedulerConfig.cosine_default(),
            # trainer=TrainerConfig.sanity(),
            # logging=LoggingConfig.sanity(),
            # checkpoint=CheckpointConfig.default(),
            # early_stopping=EarlyStoppingConfig.default(),
            # augmentations=AugmentationConfig.default(),
            # device=get_device(),
            # Diffusion-specific defaults
            method="hyper_3d",
            calculate_metric_on_test=True,
            dedup=False,
            test_sample_mult=1.1,
            filter_bad=True,
            filter_bad_path="./data/plane_problematic_shapes.txt",
            disable_wandb=False,
            normalization_factor=1,
            timesteps=500,
            use_scheduler=True,
            scheduler_step=200,
            best_model_save_path=None,
            mode="train",
            model_resume_path=None,
            sampling="ddim",
            val_fid_calculation_period=15,
            visualize_every_n_epochs=100,
            lr=0.0002,
            accumulate_grad_batches=1,
            val=ValidationConfig.default(),
            mlp_config=MLPConfig.default(),
            diff_config=DiffusionConfig.default(),
            transformer_config=TransformerConfig.default(),
            autoencoder_checkpoint=None,
            data=DataConfig.sanity(),
            optimizer=OptimizerConfig.default(),
            scheduler=SchedulerConfig.cosine_default(),
            trainer=TrainerConfig.sanity(),
            logging=LoggingConfig.sanity(),
            checkpoint=CheckpointConfig.default(),
            early_stopping=EarlyStoppingConfig.default(),
            # augmentations=AugmentationConfig.default(),
            device=get_device(),
        )
