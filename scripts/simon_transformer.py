from src.core.config import DataConfig, TransformerExperimentConfig, AugmentationConfig
from src.models.autoencoder import pl_transformer
from src.core.train import train

if __name__ == "__main__":
    config: TransformerExperimentConfig = TransformerExperimentConfig.default()
    config.data = DataConfig.small()
    config.early_stopping.min_delta = 1e-4
    config.early_stopping.patience = 100
    config.trainer.max_epochs = 1000
    config.model.latent_dim = 64
    config.model.n_tokens = 65
    config.model.num_heads = 4
    config.model.num_layers = 4
    config.model.d_model = 256  # 256 -> 4
    config.model.window_size = 16
    config.data.batch_size = 8
    config.data.sample_limit = None
    config.logging.num_samples_to_visualize = 4
    config.training.contrast = "simclr_dynamic"
    config.logging.log_every_n_steps = 1
    config.trainer.val_check_interval = 100
    # config.augmentations = AugmentationConfig.no_aug()

    # Initialize model
    model = pl_transformer.Autoencoder(config)

    train(model, config)
