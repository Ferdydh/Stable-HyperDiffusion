import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm

from src.data.utils import generate_images
from src.models.diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)

from src.core.config_diffusion import DiffusionExperimentConfig
from src.models.utils import flattened_weights_to_image_dict
from src.data.inr import INR
from src.models.diffusion.metrics import Metrics
from src.models.diffusion.transformer import Transformer


def initialize_transformer(config: DiffusionExperimentConfig) -> Transformer:
    mlp = INR(up_scale=16)
    state_dict = mlp.state_dict()
    layers = []
    layer_names = []
    for l in state_dict:
        shape = state_dict[l].shape
        layers.append(np.prod(shape))
        layer_names.append(l)

    return Transformer(
        layers,
        layer_names,
        **(config.transformer_config.as_dict()),
    )


class HyperDiffusion(pl.LightningModule):
    def __init__(
        self,
        config: DiffusionExperimentConfig,
        image_shape,
    ):
        super().__init__()
        self.model = initialize_transformer(config)
        self.config = config
        self.ae_model = None

        fake_data = torch.randn(*image_shape)

        encoded_outs = fake_data
        timesteps = config.timesteps
        betas = torch.tensor(np.linspace(1e-4, 2e-2, timesteps))
        self.image_size = encoded_outs[:1].shape

        # Initialize diffusion utiities
        self.diff = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType[config.diff_config.model_mean_type],
            model_var_type=ModelVarType[config.diff_config.model_var_type],
            loss_type=LossType[config.diff_config.loss_type],
            diff_pl_module=self,
        )

        self.training_step_outputs = []

        self.metrics = Metrics(self.device)

        # Initialize demo INR for visualization
        self.demo_inr = INR(up_scale=16).to(self.device)

    def forward(self, images):
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(images.shape[0],))
            .long()
            .to(self.device)
        )
        images = images * self.config.normalization_factor
        x_t, e = self.diff.q_sample(images, t)
        x_t = x_t.float()
        e = e.float()
        return self.model(x_t, t), e

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        if self.config.use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.scheduler_step, gamma=0.9
            )
            return [optimizer], [scheduler]
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # Extract input_data (either voxel or weight) which is the first element of the tuple
        # At the first step output first element in the dataset as a sanit check
        if self.trainer.global_step == 0:
            weights = train_batch[0].flatten().unsqueeze(0)
            self.demo_inr.eval()
            vis_dict = flattened_weights_to_image_dict(
                weights, self.demo_inr, "train/reconstruction_sanity_check", self.device
            )
            self.logger.experiment.log(vis_dict)

        # Output statistics every 100 step
        if self.trainer.global_step % 100 == 0:
            print(train_batch.shape)
            print(
                "Orig weights[0].stats",
                train_batch.min().item(),
                train_batch.max().item(),
                train_batch.mean().item(),
                train_batch.std().item(),
            )

        # Sample a diffusion timestep
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(train_batch.shape[0],))
            .long()
            .to(self.device)
        )

        # Execute a diffusion forward pass
        loss_terms = self.diff.training_losses(
            self.model,
            train_batch * self.config.normalization_factor,
            t,
            model_kwargs=None,
        )

        loss_mse = loss_terms["loss"].mean()
        self.log("train_loss", loss_mse)

        loss = loss_mse

        self.training_step_outputs.append(loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        if batch_idx == 0:
            # We are generating slightly more than ref_pcs
            number_of_samples_to_generate = int(
                self.config.val.num_samples_metrics * self.config.test_sample_mult
            )
            # Then process generated shapes
            sample_x_0s = []
            test_batch_size = 100

            for _ in tqdm(range(number_of_samples_to_generate // test_batch_size)):
                sample_x_0s.append(
                    self.diff.ddim_sample_loop(
                        self.model, (test_batch_size, *self.image_size[1:])
                    )
                )

            if number_of_samples_to_generate % test_batch_size != 0:
                sample_x_0s.append(
                    self.diff.ddim_sample_loop(
                        self.model,
                        (
                            number_of_samples_to_generate % test_batch_size,
                            *self.image_size[1:],
                        ),
                    )
                )

            sample_x_0s = torch.vstack(sample_x_0s)
            print(f"Metrics samples: {sample_x_0s.shape}")
            fake_images = generate_images(sample_x_0s, self.demo_inr, self.device)

            metrics = self.calc_metrics_2d("train", fake_images)
            for metric_name in metrics:
                self.log("train/" + metric_name, metrics[metric_name])

            metrics = self.calc_metrics_2d("val", fake_images)
            for metric_name in metrics:
                self.log("val/" + metric_name, metrics[metric_name])

    def on_train_epoch_end(self) -> None:
        epoch_average_loss = torch.stack(self.training_step_outputs).mean()
        self.log("epoch_loss", epoch_average_loss)

        # Handle 3D/4D sample generation
        if self.current_epoch % self.config.visualize_every_n_epochs == 0:
            x_0s = (
                self.diff.ddim_sample_loop(self.model, (4, *self.image_size[1:]))
                .cpu()
                .float()
            )
            x_0s = x_0s / self.config.normalization_factor
            self.demo_inr.eval()
            vis_dict = flattened_weights_to_image_dict(
                x_0s, self.demo_inr, "train/reconstruction", self.device
            )
            vis_dict["epoch"] = self.current_epoch
            self.logger.experiment.log(vis_dict)

    def calc_metrics_2d(self, split_type, fake_images):
        if split_type == "train":
            max_range = min(
                len(self.trainer.train_dataloader.dataset),
                self.config.val.num_samples_metrics,
            )
            samples = [
                self.trainer.train_dataloader.dataset[idx] for idx in range(max_range)
            ]
        elif split_type == "val":
            max_range = min(
                len(self.trainer.val_dataloaders.dataset),
                self.config.val.num_samples_metrics,
            )
            samples = [
                self.trainer.val_dataloaders.dataset[idx] for idx in range(max_range)
            ]
        elif split_type == "test":
            # TODO: add test samples
            max_range = min(
                len(self.trainer.val_dataloaders.dataset),
                self.config.val.num_samples_metrics,
            )
            samples = [
                self.trainer.val_dataloaders.dataset[idx] for idx in range(max_range)
            ]
        else:
            raise ValueError(f"Unknown split type: {split_type}")

        samples = torch.vstack(samples)

        real_images = generate_images(samples, self.demo_inr, self.device)

        metrics = self.metrics.compute_all_metrics(
            real_images.to(self.device), fake_images.to(self.device)
        )

        print("Completed metric computation for", split_type)

        return metrics
