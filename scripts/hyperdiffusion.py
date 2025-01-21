from src.core.config import (
    DataConfig,
    TransformerExperimentConfig,
)
from src.core.train_diffusion import train
from src.core.config_diffusion import DiffusionExperimentConfig


if __name__ == "__main__":
    config: DiffusionExperimentConfig = DiffusionExperimentConfig.sanity()
    # This worked for 4 samples
    config.transformer_config.n_embd = 256
    config.transformer_config.n_head = 8
    config.transformer_config.n_layer = 8

    config.logging.project_name = "hyperdiffusion"
    config.logging.run_name = "hyperdiffusion_num2"

    # Sanity
    # config.data = DataConfig.sanity()
    # config.trainer.max_epochs = 10000
    # config.visualize_every_n_epochs = 100
    # config.val_fid_calculation_period = 200

    # Test for 4
    # config.data = DataConfig.small()
    # config.data.batch_size = 4
    # config.trainer.max_epochs = 10000
    # config.data.sample_limit = 5
    # config.visualize_every_n_epochs = 100
    # config.val_fid_calculation_period = 200

    # Test for 32
    config.data = DataConfig.small()
    config.data.batch_size = 1
    config.data.sample_limit = 1
    config.trainer.max_epochs = 10000
    config.visualize_every_n_epochs = 500
    config.val_fid_calculation_period = 200

    # idk man
    # config.transformer_config.split_policy = "chunk"
    # config.transformer_config.chunk_size = 128
    config.transformer_config.n_embd = 128
    config.transformer_config.n_head = 8
    config.transformer_config.n_layer = 8

    # Test for a bigger one before full
    # config.data = DataConfig.full()
    # config.data.batch_size = 512
    # config.data.sample_limit = 565
    # config.trainer.max_epochs = 10000
    # config.visualize_every_n_epochs = 100
    # config.val_fid_calculation_period = 200

    # config.transformer_config.n_embd = 512
    # config.transformer_config.n_head = 16
    # config.transformer_config.n_layer = 12

    # Test for all 2
    # config.trainer.max_epochs = 1000
    # config.visualize_every_n_epochs = 25
    # config.val_fid_calculation_period = 50

    config.num_samples_metrics = 4
    config.early_stopping.patience = 10000

    #
    config.optimizer.lr = 2e-4
    config.scheduler.warmup_ratio = 0.05
    config.early_stopping.min_delta = 1e-6  # I want the end result to be 1e-5

    config_ae: TransformerExperimentConfig = TransformerExperimentConfig.default()
    config_ae.model.num_heads = 8
    config_ae.model.num_layers = 8
    config_ae.model.d_model = 512  # 256 -> 4
    config_ae.model.latent_dim = 8
    config_ae.model.layer_norm = False
    config_ae.model.use_mask = True
    # TODO: use one of your own checkpoints
    config.autoencoder_checkpoint = "good checkpoints/best_overfit_so_far_099_split.ckpt"

    train(config, config_ae)
