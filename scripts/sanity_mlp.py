from src.core.config import DataConfig, MLPExperimentConfig
from src.models.autoencoder import pl_mlp
from src.core.train import train

if __name__ == "__main__":
    config: MLPExperimentConfig = MLPExperimentConfig.default()
    config.data = DataConfig.sanity()
    config.early_stopping.min_delta = 1e-5
    config.early_stopping.patience = 5
    config.trainer.max_epochs = 10
    # config.data.batch_size = 1

    # Initialize model
    model = pl_mlp.Autoencoder(config)

    train(model, config)
