from src.core.config import DataConfig, TransformerExperimentConfig
from src.models.autoencoder import pl_transformer
from src.core.train import train

if __name__ == "__main__":
    config: TransformerExperimentConfig = TransformerExperimentConfig.default()
    # config.data = DataConfig.sanity()
    # config.early_stopping.min_delta = 1e-5
    # config.early_stopping.patience = 100
    # config.trainer.max_epochs = 1000
    # config.augmentations = AugmentationConfig.no_aug()
    config.logging.sample_every_n_epochs = 50

    # config.model.d_model = 64
    # config.model.latent_dim = 8

    config.data = DataConfig.small()
    config.early_stopping.min_delta = 1e-5
    config.early_stopping.patience = 400
    config.trainer.max_epochs = 2000

    #
    config.model.latent_dim = 64
    config.model.num_heads = 4
    config.model.num_layers = 4
    config.model.num_layers = 16
    config.model.d_model = 256  # 256 -> 4
    # config.model.window_size = 65  # no window
    # config.model.window_size = 16
    config.data.batch_size = 8

    config.data.split_ratio = 0.8

    # Is tied together
    # config.trainer.val_check_interval = None
    config.data.sample_limit = 10

    config.logging.num_samples_to_visualize = 3
    config.logging.log_every_n_steps = 10

    config.scheduler.eta_min = 1e-4
    config.scheduler.T_max = 2000
    config.optimizer.lr = 1e-3

    # from dataclasses import asdict

    # print(asdict(config))
    # pass

    # Initialize model
    model = pl_transformer.Autoencoder(config)

    train(model, config)
