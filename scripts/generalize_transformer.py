import torch
from src.core.config import DataConfig, TransformerExperimentConfig
from src.models.autoencoder import pl_transformer
from src.core.train import train

if __name__ == "__main__":
    config: TransformerExperimentConfig = TransformerExperimentConfig.default()

    # Logging
    # config.logging.sample_every_n_epochs = 50
    config.logging.num_samples_to_visualize = 4
    config.logging.log_every_n_steps = 10
    config.logging.sample_every_n_epochs = 50

    config.early_stopping.min_delta = 1e-6  # I want the end result to be 1e-5
    # config.early_stopping.patience = 400
    # config.trainer.max_epochs = 2000

    #
    config.model.num_heads = 8
    config.model.num_layers = 8

    # config.model.d_model = 64
    # config.model.latent_dim = 8

    config.model.d_model = 512  # 256 -> 4
    config.model.latent_dim = 8
    config.model.dropout = 0.01

    config.model.beta = 0.1
    config.model.recon_scale = 1.0e4
    config.model.use_mask = True
    config.model.noise = 1e-4
    config.model.layer_norm = True

    config.early_stopping.patience = 900
    config.trainer.max_epochs = 1000
    config.scheduler.warmup_ratio = 0.05

    # data
    config.data = DataConfig.small()
    config.data.batch_size = 32
    config.data.split_ratio = 0.8
    config.data.sample_limit = None
    config.data.num_workers = 4
    config.data.load_from_txt = False
    #config.data = DataConfig.full()

    # config.scheduler.eta_min = 1e-4
    # config.scheduler.T_max = 1000
    config.optimizer.lr = 1e-4
    config.optimizer.weight_decay = 1e-4

    # from dataclasses import asdict

    # print(asdict(config))
    # pass

    # Initialize model
    model = pl_transformer.Autoencoder(config)
    #state_dict = torch.load("logs/checkpoints/last-v22.ckpt")
    #model.load_state_dict(state_dict["state_dict"])

    train(model, config)
