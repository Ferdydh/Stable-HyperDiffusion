import copy
import os

import numpy as np
import pytorch_lightning as pl
import torch
from itertools import islice
import random
#import trimesh
#from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import wandb
from src.data.utils import generate_images
from src.models.diffusion.gaussian_diffusion import (GaussianDiffusion, LossType,
                                          ModelMeanType, ModelVarType)

from src.core.config import DiffusionExperimentConfig
from src.models.utils import (
    weights_to_image_dict, flattened_weights_to_image_dict
)
from src.data.data_converter import flattened_weights_to_weights
from src.data.inr import INR

from src.models.diffusion.metrics import Metrics
#from evaluation_metrics_3d import compute_all_metrics, compute_all_metrics_4d
#from hd_utils import (Config, calculate_fid_3d, generate_mlp_from_weights,
#                      render_mesh, render_meshes)
#from siren import sdf_meshing
#from siren.dataio import anime_read
#from siren.experiment_scripts.test_sdf import SDFDecoder


class HyperDiffusion(pl.LightningModule):
    def __init__(
        #self, model, train_dt, val_dt, test_dt, mlp_kwargs, image_shape, method, config: DiffusionExperimentConfig
        self, model, config: DiffusionExperimentConfig, image_shape
    ):
        super().__init__()
        self.model = model
        self.config = config
        #self.method = method
        #self.mlp_kwargs = mlp_kwargs
        #self.val_dt = val_dt
        #self.train_dt = train_dt
        #self.test_dt = test_dt
        self.ae_model = None

        fake_data = torch.randn(*image_shape)

        encoded_outs = fake_data
        #print("encoded_outs.shape", encoded_outs.shape)
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

    #def grid_to_mesh(self, grid):
    #    grid = np.where(grid > 0, True, False)
    #    vox_grid = trimesh.voxel.VoxelGrid(grid)
    #    try:
    #        vox_grid = vox_grid.marching_cubes
    #    except:
    #        return vox_grid.as_boxes()
    #    vert = vox_grid.vertices
    #    if len(vert) == 0:
    #        return vox_grid
    #    vert /= grid.shape[-1]
    #    vert = 2 * vert - 1
    #    vox_grid.vertices = vert
    #    return vox_grid

    def training_step(self, train_batch, batch_idx):
        # Extract input_data (either voxel or weight) which is the first element of the tuple
        #input_data = train_batch[0].unsqueeze(0)
        #if self.current_epoch == 0:
        #    train_batch = next(islice(self.trainer.train_dataloader, batch_idx, batch_idx+1))  # Get the x-th batch

        # At the first step output first element in the dataset as a sanit check
        if self.trainer.global_step == 0:
            weights = train_batch[0].flatten().unsqueeze(0)
            self.demo_inr.eval()
            #v = weights_to_image_dict(
                #[flattened_weights_to_weights(weights, self.demo_inr)],
            #    self.demo_inr,
             #   "train/reconstruction_sanity_check",
             #   self.device
            #)
            vis_dict = flattened_weights_to_image_dict(
                weights,
                self.demo_inr,
                "train/reconstruction_sanity_check",
                self.device
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
            number_of_samples_to_generate = int(self.config.val.num_samples_metrics * self.config.test_sample_mult)
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
            #print(
            #    "x_0s[0].stats",
            #    x_0s.min().item(),
            #    x_0s.max().item(),
            #    x_0s.mean().item(),
            #    x_0s.std().item(),
            #)
            
            #meshes, sdfs = self.generate_meshes(x_0s, None, res=512)
            #for mesh in meshes:
            #    mesh.vertices *= 2
            self.demo_inr.eval()
            vis_dict = flattened_weights_to_image_dict(
                x_0s,
                self.demo_inr,
                "train/reconstruction",
                self.device
            )
            vis_dict["epoch"] = self.current_epoch
            self.logger.experiment.log(vis_dict)

            #print(
            #    "sdfs.stats",
                #sdfs.min().item(),
                #sdfs.max().item(),
                #sdfs.mean().item(),
                #sdfs.std().item(),
            #)

            #out_imgs = render_meshes(meshes)
            #self.logger.log_image(
            #    "generated_renders", out_imgs, step=self.current_epoch
            #)


    #def print_summary(self, flat, func):
    #    var = func(flat, dim=0)
    #    print(
    #        var.shape,
     #       var.mean().item(),
    #        var.std().item(),
     #       var.min().item(),
     #       var.max().item(),
      #  )
      #  print(var.shape, func(flat))



    def calc_metrics_2d(self, split_type, fake_images):
        if split_type == "train":
            max_range = min(len(self.trainer.train_dataloader.dataset), self.config.val.num_samples_metrics)
            samples = [self.trainer.train_dataloader.dataset[idx] for idx in range(max_range)]
        elif split_type == "val":
            max_range = min(len(self.trainer.val_dataloaders.dataset), self.config.val.num_samples_metrics)
            samples = [self.trainer.val_dataloaders.dataset[idx] for idx in range(max_range)]
        elif split_type == "test":
            # TODO: add test samples
            max_range = min(len(self.trainer.val_dataloaders.dataset), self.config.val.num_samples_metrics)
            samples = [self.trainer.val_dataloaders.dataset[idx] for idx in range(max_range)]
        else:
            raise ValueError(f"Unknown split type: {split_type}")
        
        samples = torch.vstack(samples)
    
        real_images = generate_images(samples, self.demo_inr, self.device)

        metrics = self.metrics.compute_all_metrics(
            real_images.to(self.device),
            fake_images.to(self.device)
        )

        print("Completed metric computation for", split_type)

        return metrics
    
    """
    def test_step(self, *args, **kwargs):
        if self.config.calculate_metric_on_test:
            metrics = self.calc_metrics_2d("test")
            print("test", metrics)
            for metric_name in metrics:
                self.log("test/" + metric_name, metrics[metric_name])

        
        # If it's HyperDiffusion, let's calculate some statistics on training dataset
        #elif self.method == "hyper_3d":
        x_0s = []
        for i, img in enumerate(self.train_dt):
            x_0s.append(img[0])
        x_0s = torch.stack(x_0s).to(self.device)
        flat = x_0s.view(len(x_0s), -1)
        # return
        print(x_0s.shape, flat.shape)
        print("Variance With zero-padding")
        self.print_summary(flat, torch.var)
        print("Variance Without zero-padding")
        #self.print_summary(flat[:, : Config.get("curr_weights")], torch.var)

        print("Mean With zero-padding")
        self.print_summary(flat, torch.mean)
        print("Mean Without zero-padding")
        #self.print_summary(flat[:, : Config.get("curr_weights")], torch.mean)

        stdev = x_0s.flatten().std(unbiased=True).item()
        oai_coeff = (
            0.538 / stdev
        )  # 0.538 is the variance of ImageNet pixels scaled to [-1, 1]
        print(f"Standard Deviation: {stdev}")
        print(f"OpenAI Coefficient: {oai_coeff}")

        # Then, sampling some new shapes -> outputting and rendering them
        x_0s = self.diff.ddim_sample_loop(
            self.model, (16, *self.image_size[1:]), clip_denoised=False
        )
        x_0s = x_0s / self.cfg.normalization_factor

        print(
            "x_0s[0].stats",
            x_0s.min().item(),
            x_0s.max().item(),
            x_0s.mean().item(),
            x_0s.std().item(),
        )
        out_pc_imgs = []

        
        # Handle 3D generation
        #else:
        out_imgs = []
        os.makedirs(f"gen_meshes/{wandb.run.name}")
        for x_0 in tqdm(x_0s):
            mesh, _ = self.generate_meshes(x_0.unsqueeze(0), None, res=700)
            mesh = mesh[0]
            if len(mesh.vertices) == 0:
                continue
            mesh.vertices *= 2
            mesh.export(f"gen_meshes/{wandb.run.name}/mesh_{len(out_imgs)}.obj")

            # Scaling the chairs down so that they fit in the camera
            if self.cfg.dataset == "03001627":
                mesh.vertices *= 0.7
            img, _ = render_mesh(mesh)

            if len(mesh.vertices) > 0:
                pc = torch.tensor(mesh.sample(2048))
            else:
                print("Empty mesh")
                pc = torch.zeros(2048, 3)
            pc_img, _ = render_mesh(pc)
            out_imgs.append(img)
            out_pc_imgs.append(pc_img)

        self.logger.log_image(
            "generated_renders", out_imgs, step=self.current_epoch
        )
        self.logger.log_image(
            "generated_renders_pc", out_pc_imgs, step=self.current_epoch
        )

        """