from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import multiprocessing
import random
import pytorch_lightning as pl

from src.core.config import (
    BaseExperimentConfig,
    MLPExperimentConfig,
)
from src.data.data_converter import weights_to_flattened_weights, weights_to_tokens
from src.data.utils import get_files_from_selectors

# Ensure 'spawn' start method is set globally for multiprocessing
multiprocessing.set_start_method("spawn", force=True)


class INRDataset(Dataset):
    def __init__(self, files, device, is_for_mlp):
        self.files = files
        self.device = device
        self.is_for_mlp = is_for_mlp

    def __getitem__(self, index):
        file_path = self.files[index]
        state_dict = torch.load(file_path, map_location=self.device, weights_only=True)

        if self.is_for_mlp:
            return weights_to_flattened_weights(state_dict)

        # For transformer, convert weights to tokens
        tokens, masks, pos = weights_to_tokens(
            state_dict, tokensize=0, device=self.device
        )

        return tokens, masks, pos

    def get_state_dict(self, index):
        return torch.load(
            self.files[index], map_location=self.device, weights_only=True
        )

    def __len__(self):
        return len(self.files)


class DataHandler(pl.LightningDataModule):
    def __init__(
        self,
        config: BaseExperimentConfig,
    ):
        super().__init__()

        self.config = config

        self.split_ratio = config.data.split_ratio
        self.data_path = os.path.join(os.getcwd(), config.data.data_path)

        # this determines if the data is flattened or tokenized
        self.is_for_mlp = isinstance(config, MLPExperimentConfig)

        self.files = get_files_from_selectors(self.data_path, [config.data.selector])

        if self.split_ratio < 0.5:
            print("Warning: Split ratio is less than 0.5.")

        # Limit the number of samples if specified
        if (
            self.config.data.sample_limit is not None
            and self.config.data.sample_limit < len(self.files)
        ):
            self.files = random.sample(self.files, self.config.data.sample_limit)

        if len(self.files) == 0:
            raise ValueError(
                f"No files found matching the selection criteria in {self.data_path}"
            )

    def setup(self, stage: Optional[str] = None):
        # Create split datasets
        if stage == "fit" or stage is None:
            # Calculate split sizes
            train_size = int(len(self.files) * self.split_ratio)
            val_size = len(self.files) - train_size

            if train_size == len(self.files) or val_size == 0 or train_size == 0:
                # This should only happen if we have sample_limit=1 or split_ratio=1.0
                train_files = self.files
                val_files = self.files
            else:
                train_files, val_files = random_split(
                    self.files, [train_size, val_size]
                )

            self.train_dataset = INRDataset(
                files=train_files, device=self.config.device, is_for_mlp=self.is_for_mlp
            )
            self.val_dataset = INRDataset(
                files=val_files, device=self.config.device, is_for_mlp=self.is_for_mlp
            )
            print(f"Train size: {len(self.train_dataset)}")
            print(f"Val size: {len(self.val_dataset)}")

            # Check if there are any overlapping files between train and val
            overlapping_files = set(self.train_dataset.files).intersection(
                set(self.val_dataset.files)
            )
            if overlapping_files:
                print(
                    "Warning: There are overlapping files between train and val datasets."
                )
                print("Overlapping files: ", len(overlapping_files))
            else:
                print("No overlapping files between train and val datasets.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=True,
            persistent_workers=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=False,
            persistent_workers=True,
            drop_last=False,
        )

    def get_state_dict(self, index):
        return self.train_dataset.get_state_dict(index)
