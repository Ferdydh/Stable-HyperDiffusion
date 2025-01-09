from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Literal, Union, Dict, Any

import torch


DATA_PATH = "mnist-inrs"


class DatasetType(Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"

    @classmethod
    def from_str(cls, value: Optional[str]) -> Optional["DatasetType"]:
        """Convert string to DatasetType, returning None if input is None."""
        if value is None:
            return None
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Invalid dataset type: {value}")


@dataclass
class DataSelector:
    """
    Flexible data selection criteria.

    Examples:
        # Select all data (both MNIST and CIFAR10)
        DataSelector(dataset_type=None)
        DataSelector()

        # Select all MNIST
        DataSelector(dataset_type=DatasetType.MNIST)

        # Select only MNIST class 1
        DataSelector(dataset_type=DatasetType.MNIST, class_label=1)

        # Select MNIST class 1 with specific ID
        DataSelector(dataset_type=DatasetType.MNIST, class_label=1, sample_id=100)
    """

    dataset_type: Optional[DatasetType]
    class_label: Optional[Union[int, str]] = None
    sample_id: Optional[int] = None


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Base Configs
@dataclass
class BaseModelConfig:
    """Base configuration for all models"""

    pass


@dataclass
class MLPModelConfig(BaseModelConfig):
    input_dim: int
    output_dim: int
    hidden_dim: int
    z_dim: int
    activation = torch.nn.ReLU

    def __post_init__(self):
        if self.z_dim >= self.hidden_dim:
            raise ValueError("z_dim must be smaller than hidden_dim")

    @classmethod
    def default(cls) -> "MLPModelConfig":
        return cls(
            input_dim=1185,
            output_dim=1185,
            hidden_dim=1185,
            z_dim=64,
        )


@dataclass
class TransformerModelConfig(BaseModelConfig):
    max_positions: List[int]
    num_layers: int
    d_model: int
    dropout: float
    window_size: int
    num_heads: int
    input_dim: int
    n_tokens: int
    latent_dim: int
    output_dim: int
    projection_dim: int

    @classmethod
    def default(cls) -> "TransformerModelConfig":
        return cls(
            max_positions=[100, 10, 40],
            num_layers=8,
            d_model=1024,
            dropout=0.0,
            window_size=32,
            num_heads=8,
            input_dim=33,
            n_tokens=65,
            latent_dim=128,
            output_dim=30,
            projection_dim=128,
        )


@dataclass
class DataConfig:
    data_path: str
    selector: "DataSelector"
    batch_size: int = 32
    num_workers: int = 4
    sample_limit: Optional[int] = None
    split_ratio: float = 0.9  # Train-Val split ratio of 90-10%

    @classmethod
    def sanity(cls) -> "DataConfig":
        return cls(
            data_path=DATA_PATH,
            selector=DataSelector(
                dataset_type=DatasetType.MNIST, class_label=2, sample_id=1096
            ),
            batch_size=1,
            sample_limit=1,
        )

    @classmethod
    def small(cls) -> "DataConfig":
        return cls(
            data_path=DATA_PATH,
            selector=DataSelector(dataset_type=DatasetType.MNIST, class_label=2),
            batch_size=2,
            sample_limit=2,
        )

    @classmethod
    def full(cls) -> "DataConfig":
        return cls(
            data_path=DATA_PATH,
            selector=DataSelector(dataset_type=DatasetType.MNIST, class_label=2),
            batch_size=32,
        )


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-5
    eps: float = 1e-8

    @classmethod
    def default(cls) -> "OptimizerConfig":
        return cls(
            name="adamw",
            lr=1e-4,
            weight_decay=3e-9,
        )


@dataclass
class SchedulerConfig:
    name: str
    T_max: Optional[int] = None
    eta_min: Optional[float] = None

    @classmethod
    def cosine_default(cls) -> "SchedulerConfig":
        return cls(
            name="cosine",
            T_max=1000,
            eta_min=1e-6,
        )


@dataclass
class TrainerConfig:
    max_epochs: int
    precision: Literal[16, 32, 64]
    gradient_clip_val: float
    accumulate_grad_batches: int
    val_check_interval: float
    log_every_n_steps: int

    @classmethod
    def sanity(cls) -> "TrainerConfig":
        return cls(
            max_epochs=10,
            precision=32,
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            val_check_interval=1.0,
            log_every_n_steps=1,
        )

    @classmethod
    def default(cls) -> "TrainerConfig":
        return cls(
            max_epochs=100,
            precision=32,
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            val_check_interval=1.0,
            log_every_n_steps=50,
        )


@dataclass
class LoggingConfig:
    project_name: str
    save_dir: str
    num_samples_to_visualize: int
    sample_every_n_epochs: int
    log_every_n_steps: int
    run_name: Optional[str] = None
    log_model: bool = False

    @classmethod
    def sanity(cls) -> "LoggingConfig":
        return cls(
            project_name="inr-autoencoder",
            save_dir="logs",
            num_samples_to_visualize=1,
            sample_every_n_epochs=1,
            log_every_n_steps=1,
        )

    @classmethod
    def default(cls) -> "LoggingConfig":
        return cls(
            project_name="inr-autoencoder",
            save_dir="logs",
            num_samples_to_visualize=3,
            sample_every_n_epochs=50,
            log_every_n_steps=50,
        )


@dataclass
class CheckpointConfig:
    dirpath: str
    filename: str
    monitor: str
    mode: Literal["min", "max"]
    save_last: bool
    save_top_k: int

    @classmethod
    def default(cls) -> "CheckpointConfig":
        return cls(
            dirpath="checkpoints",
            filename="autoencoder-{epoch:02d}-{val_loss:.2f}",
            monitor="val/loss",
            mode="min",
            save_last=True,
            save_top_k=3,
        )


@dataclass
class EarlyStoppingConfig:
    monitor: str
    min_delta: float
    patience: int
    mode: Literal["min", "max"]

    @classmethod
    def sanity(cls) -> "EarlyStoppingConfig":
        return cls(
            monitor="val/loss",
            min_delta=1e-4,
            patience=5,
            mode="min",
        )

    @classmethod
    def default(cls) -> "EarlyStoppingConfig":
        return cls(
            monitor="val/loss",
            min_delta=1e-5,
            patience=200,
            mode="min",
        )


@dataclass
class TrainingConfig:
    """Contrastive learning configuration for transformer"""

    gamma: float
    reduction: str
    temperature: float
    contrast: str
    z_var_penalty: float

    @classmethod
    def default(cls) -> "TrainingConfig":
        return cls(
            gamma=0.05,
            reduction="mean",
            temperature=0.1,
            contrast="simclr",
            z_var_penalty=0.0,
        )


@dataclass
class AugmentationConfig:
    """Contrastive learning configuration for transformer"""

    permutation_number_train: int
    permutation_number_val: int
    view_1_canon_train: bool
    view_1_canon_val: bool
    view_2_canon_train: bool
    view_2_canon_val: bool
    add_noise_view_1_train: float
    add_noise_view_1_val: float
    add_noise_view_2_train: float
    add_noise_view_2_val: float
    erase_augment_view_1_train: float
    erase_augment_view_2_train: float
    erase_augment_view_1_val: float
    erase_augment_view_2_val: float
    multi_windows_train: float
    apply_augmentations: bool

    @classmethod
    def default(cls) -> "AugmentationConfig":
        return cls(
            permutation_number_train=5,
            permutation_number_val=5,
            view_1_canon_train=False,
            view_1_canon_val=True,
            view_2_canon_train=True,
            view_2_canon_val=False,
            add_noise_view_1_train=0.1,
            add_noise_view_1_val=0.0,
            add_noise_view_2_train=0.1,
            add_noise_view_2_val=0.0,
            erase_augment_view_1_train=None,
            erase_augment_view_2_train=None,
            erase_augment_view_1_val=None,
            erase_augment_view_2_val=None,
            multi_windows_train=None,
            apply_augmentations=True,
        )
    
    @classmethod
    def no_aug(cls) -> "AugmentationConfig":
        return cls(
            permutation_number_train=5,
            permutation_number_val=5,
            view_1_canon_train=False,
            view_1_canon_val=True,
            view_2_canon_train=True,
            view_2_canon_val=False,
            add_noise_view_1_train=0.1,
            add_noise_view_1_val=0.0,
            add_noise_view_2_train=0.1,
            add_noise_view_2_val=0.0,
            erase_augment_view_1_train=None,
            erase_augment_view_2_train=None,
            erase_augment_view_1_val=None,
            erase_augment_view_2_val=None,
            multi_windows_train=None,
            apply_augmentations=False,
        )


@dataclass
class BaseExperimentConfig:
    """Base configuration for all experiments"""

    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    trainer: TrainerConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    early_stopping: EarlyStoppingConfig
    augmentations: AugmentationConfig
    device: torch.device


@dataclass
class MLPExperimentConfig(BaseExperimentConfig):
    model: MLPModelConfig

    @classmethod
    def default(cls) -> "MLPExperimentConfig":
        return cls(
            data=DataConfig.sanity(),
            model=MLPModelConfig.default(),
            optimizer=OptimizerConfig.default(),
            scheduler=SchedulerConfig.cosine_default(),
            trainer=TrainerConfig.sanity(),
            logging=LoggingConfig.sanity(),
            checkpoint=CheckpointConfig.default(),
            early_stopping=EarlyStoppingConfig.default(),
            augmentations=None,
            device=get_device(),
        )


@dataclass
class TransformerExperimentConfig(BaseExperimentConfig):
    model: TransformerModelConfig
    training: TrainingConfig

    @classmethod
    def default(cls) -> "TransformerExperimentConfig":
        return cls(
            data=DataConfig.sanity(),
            model=TransformerModelConfig.default(),
            training=TrainingConfig.default(),
            optimizer=OptimizerConfig.default(),
            scheduler=SchedulerConfig.cosine_default(),
            trainer=TrainerConfig.sanity(),
            logging=LoggingConfig.sanity(),
            checkpoint=CheckpointConfig.default(),
            early_stopping=EarlyStoppingConfig.default(),
            augmentations=AugmentationConfig.default(),
            device=get_device(),
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
            move=False
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
            model_mean_type="START_X",
            model_var_type="FIXED_LARGE",
            loss_type="MSE"
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
            condition="no"
        )
    
    @classmethod
    def get_dict(cls, transformer_config: "TransformerConfig") -> Dict[str, Any]:
        return {
            "n_embd": transformer_config.n_embd,
            "n_layer": transformer_config.n_layer,
            "n_head": transformer_config.n_head,
            "split_policy": transformer_config.split_policy,
            "use_global_residual": transformer_config.use_global_residual,
            "condition": transformer_config.condition
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
            num_samples_visualization = 4,
            visualize_every_n_epochs = 20
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
            #data=DataConfig.sanity(),
            #optimizer=OptimizerConfig.default(),
            #scheduler=SchedulerConfig.cosine_default(),
            #trainer=TrainerConfig.sanity(),
            #logging=LoggingConfig.sanity(),
            #checkpoint=CheckpointConfig.default(),
            #early_stopping=EarlyStoppingConfig.default(),
            #augmentations=AugmentationConfig.default(),
            #device=get_device(),
            
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
            augmentations=AugmentationConfig.default(),
            device=get_device(),
        )