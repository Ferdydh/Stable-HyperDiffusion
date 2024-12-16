from typing import List
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import multiprocessing
import random
import pytorch_lightning as pl

from src.core.config import (
    DataSelector,
    DatasetType,
    MLPExperimentConfig,
    TransformerExperimentConfig,
)
from src.data.utils import weights_to_flattened_weights, weights_to_tokens

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


def collate_state_dicts_as_list(batch):
    """
    Custom collate function to return a batch as a list of state_dicts.
    Args:
        batch (list): A list of state_dicts from the Dataset.
    Returns:
        list: The batch as a list of state_dicts.
    """
    return batch


class DataHandler(pl.LightningDataModule):
    def __init__(
        self,
        config: MLPExperimentConfig | TransformerExperimentConfig,
    ):
        super().__init__()

        self.config = config

        self.split_ratio = config.data.split_ratio
        self.data_path = os.path.join(os.getcwd(), config.data.data_path)

        # this determines if the data is flattened or tokenized
        self.is_for_mlp = isinstance(config, MLPExperimentConfig)

        self.files = self._get_files_from_selectors([config.data.selector])

        if len(self.files) == 0:
            raise ValueError(
                f"No files found matching the selection criteria in {self.data_path}"
            )

        if config.data.sample_limit is not None:
            if config.data.sample_limit > len(self.files):
                config.data.sample_limit = len(self.files)
            self.files = random.sample(self.files, config.data.sample_limit)

        if config.data.sample_limit is not None and config.data.sample_limit == 1:
            # Only training dataloader matters
            self._create_single_split()
        else:
            self._create_train_val_test_split()

    def _get_files_from_selectors(self, selectors: List[DataSelector]) -> List[str]:
        """Get all files matching the given selectors."""
        all_files = []

        for selector in selectors:
            matching_files = []

            # List all files in the data directory
            for f in os.listdir(self.data_path):
                # If no dataset type specified, match both MNIST and CIFAR10
                if selector.dataset_type is None:
                    if not (f.startswith("mnist_png_") or f.startswith("cifar10_png_")):
                        continue
                # For MNIST: match mnist_png_(train|test)_digit_id pattern
                elif selector.dataset_type == DatasetType.MNIST:
                    if not f.startswith("mnist_png_"):
                        continue
                # For CIFAR10: match cifar10_png_train_class_id pattern
                elif selector.dataset_type == DatasetType.CIFAR10:
                    if not f.startswith("cifar10_png_train"):
                        continue

                # If class label is specified, match only that class/digit
                if selector.class_label is not None:
                    if f"_{selector.class_label}_" not in f:
                        continue

                # If sample_id is specified, match that specific ID
                if selector.sample_id is not None:
                    if not f.endswith(f"_{selector.sample_id}"):
                        continue

                # Add the full path to the model file
                model_path = os.path.join(
                    os.path.join(os.path.join(self.data_path, f), "checkpoints"),
                    "model_final.pth",
                )
                if os.path.exists(model_path):
                    matching_files.append(model_path)

            all_files.extend(matching_files)

        return sorted(
            list(set(all_files))
        )  # Remove duplicates and sort for consistency

    def _create_single_split(self):
        """Create a single dataset for all splits (used with sample_limit)."""
        self.train_dataset = self.val_dataset = self.test_dataset = INRDataset(
            self.files, device=self.config.device, is_for_mlp=self.is_for_mlp
        )

    def _create_train_val_test_split(self):
        """Create train/val/test splits based on split ratio."""
        train_len = int(len(self.files) * self.split_ratio[0] / 100)
        val_len = int(len(self.files) * self.split_ratio[1] / 100)
        test_len = len(self.files) - train_len - val_len

        train_files, val_files, test_files = random_split(
            self.files, [train_len, val_len, test_len]
        )

        print(
            f"Train: {len(train_files)} Val: {len(val_files)} Test: {len(test_files)}"
        )

        self.train_dataset = INRDataset(
            train_files, device=self.config.device, is_for_mlp=self.is_for_mlp
        )
        self.val_dataset = INRDataset(
            val_files, device=self.config.device, is_for_mlp=self.is_for_mlp
        )
        self.test_dataset = INRDataset(
            test_files, device=self.config.device, is_for_mlp=self.is_for_mlp
        )

    def train_dataloader(self):
        collate_fn = collate_state_dicts_as_list

        if self.is_for_mlp:
            collate_fn = None
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        collate_fn = collate_state_dicts_as_list

        if self.is_for_mlp:
            collate_fn = None

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=False,
            persistent_workers=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def test_dataloader(self):
        collate_fn = collate_state_dicts_as_list

        if self.is_for_mlp:
            collate_fn = None
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=False,
            persistent_workers=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def get_state_dict(self, index):
        return self.train_dataset.get_state_dict(index)
