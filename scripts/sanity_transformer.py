from src.core.config import DataConfig, TransformerExperimentConfig
from src.models.autoencoder import pl_transformer
from src.core.train import train

if __name__ == "__main__":
    config: TransformerExperimentConfig = TransformerExperimentConfig.default()
    config.data = DataConfig.sanity()
    config.early_stopping.min_delta = 1e-4
    config.early_stopping.patience = 30
    config.trainer.max_epochs = 300
    # config.augmentations = AugmentationConfig.no_aug()

    # Initialize model
    model = pl_transformer.Autoencoder(config)

    train(model, config)
