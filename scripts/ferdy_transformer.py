from src.core.config import DataConfig, TransformerExperimentConfig
from src.models.autoencoder import pl_transformer
from src.core.train import train

if __name__ == "__main__":
    config: TransformerExperimentConfig = TransformerExperimentConfig.default()

    # Logging
    # config.logging.sample_every_n_epochs = 50
    config.logging.sample_every_n_epochs = 1
    config.logging.num_samples_to_visualize = 3
    config.logging.log_every_n_steps = 10

    config.early_stopping.min_delta = 1e-6  # I want the end result to be 1e-5
    # config.early_stopping.patience = 400
    # config.trainer.max_epochs = 2000

    #
    config.model.num_heads = 8
    config.model.num_layers = 8

    # config.model.d_model = 64
    # config.model.latent_dim = 8

    config.model.d_model = 128  # 256 -> 4
    config.model.latent_dim = 32

    config.early_stopping.patience = 900
    config.trainer.max_epochs = 1000
    # config.scheduler.warmup_ratio = 0.05

    # data
    config.data = DataConfig.small()
    config.data.batch_size = 8
    config.data.split_ratio = 0.8
    config.data.sample_limit = 10
    config.data = DataConfig.full()

    # config.scheduler.eta_min = 1e-4
    # config.scheduler.T_max = 1000
    config.optimizer.lr = 1e-4

    # from dataclasses import asdict

    # print(asdict(config))
    # pass

    # Initialize model
    model = pl_transformer.Autoencoder(config)

    train(model, config)
