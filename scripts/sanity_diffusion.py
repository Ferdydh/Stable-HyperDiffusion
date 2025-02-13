from src.core.config import (
    DataConfig,
    TransformerExperimentConfig,
)
from src.core.train_diffusion import train
from src.core.config_diffusion import DiffusionExperimentConfig

if __name__ == "__main__":
    config: DiffusionExperimentConfig = DiffusionExperimentConfig.sanity()
    config.transformer_config.n_embd = 64
    # config.transformer_config.n_embd = 128
    config.transformer_config.n_head = 8
    config.transformer_config.n_layer = 8

    config.logging.project_name = "hyperdiffusion"
    config.logging.run_name = "hyperdiffusion_num2"
    config.data = DataConfig.small()
    config.data.batch_size = 4
    config.data.sample_limit = 4
    config.trainer.max_epochs = 50
    # config.trainer.max_epochs = 500
    config.num_samples_metrics = 4
    config.visualize_every_n_epochs = 10
    config.val_fid_calculation_period = 10

    config_ae = TransformerExperimentConfig.default()

    train(config, config_ae)
