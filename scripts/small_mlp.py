from src.core.config import DataConfig, MLPExperimentConfig, LoggingConfig
from src.models.autoencoder import pl_mlp
from src.core.train import train

if __name__ == "__main__":
    config: MLPExperimentConfig = MLPExperimentConfig.default()
    config.data = DataConfig.small()
    config.early_stopping.min_delta = 1e-5
    config.early_stopping.patience = 200
    config.trainer.max_epochs = 1000
    config.logging = LoggingConfig.sanity()
    config.data.batch_size = 8
    config.data.sample_limit = 8
    

    # Initialize model
    model = pl_mlp.Autoencoder(config)

    train(model, config)
