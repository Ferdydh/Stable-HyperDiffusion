from src.core.config import DataConfig, MLPExperimentConfig
from src.models.autoencoder import pl_transformer
from src.core.train import train

if __name__ == "__main__":
    config: MLPExperimentConfig = MLPExperimentConfig.default()
    config.data = DataConfig.sanity(False)
    config.early_stopping.min_delta = 1e-4
    config.early_stopping.patience = 30
    config.trainer.max_epochs = 300

    # Initialize model
    model = pl_transformer.Autoencoder(config)

    train(model, config)
