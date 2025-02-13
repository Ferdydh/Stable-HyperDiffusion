from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Literal, Union

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
    beta: float = 0.1
    recon_scale: float = 1.0e-4
    use_mask: bool = True
    noise: float = 1e-4
    layer_norm: bool = True

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
    load_from_txt: bool = False

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
    
    @classmethod
    def all_digits(cls) -> "DataConfig":
        return cls(
            data_path=DATA_PATH,
            selector=DataSelector(dataset_type=DatasetType.MNIST),
            batch_size=64,
        )


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    eps: float = 1e-8

    @classmethod
    def default(cls) -> "OptimizerConfig":
        return cls(
            name="adamw",
            lr=1e-4,
            weight_decay=0.01,
        )


@dataclass
class SchedulerConfig:
    warmup_ratio: float
    # name: str
    #T_max: Optional[int] = None
    #eta_min: Optional[float] = None

    @classmethod
    def cosine_default(cls) -> "SchedulerConfig":
        return cls(
            warmup_ratio=0.1,
            # name="cosine",
            # TODO: fix this haha. this one is for cosine annealing, the one on top is for other schedulers
            #T_max=1000,
            #eta_min=1e-6,
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
            dirpath="logs/checkpoints",
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
class BaseExperimentConfig:
    """Base configuration for all experiments"""

    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    trainer: TrainerConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    early_stopping: EarlyStoppingConfig
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
            device=get_device(),
        )


@dataclass
class TransformerExperimentConfig(BaseExperimentConfig):
    model: TransformerModelConfig

    @classmethod
    def default(cls) -> "TransformerExperimentConfig":
        return cls(
            data=DataConfig.sanity(),
            model=TransformerModelConfig.default(),
            optimizer=OptimizerConfig.default(),
            scheduler=SchedulerConfig.cosine_default(),
            trainer=TrainerConfig.sanity(),
            logging=LoggingConfig.sanity(),
            checkpoint=CheckpointConfig.default(),
            early_stopping=EarlyStoppingConfig.default(),
            device=get_device(),
        )
