from enum import Enum
from typing import Union, List, Optional, Dict, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import multiprocessing
import random

# Ensure 'spawn' start method is set globally for multiprocessing
multiprocessing.set_start_method("spawn", force=True)


class DatasetType(Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"


@dataclass
class DataSelector:
    """
    Flexible data selection criteria.

    Examples:
        # Select all MNIST
        DataSelector(dataset_type=DatasetType.MNIST)

        # Select only MNIST class 1
        DataSelector(dataset_type=DatasetType.MNIST, class_label=1)

        # Select MNIST class 1 with specific ID
        DataSelector(dataset_type=DatasetType.MNIST, class_label=1, sample_id=100)
    """

    dataset_type: DatasetType
    class_label: Optional[Union[int, str]] = None
    sample_id: Optional[int] = None


class IRNDataset(Dataset):
    def __init__(self, files, device):
        self.files = files
        self.device = device

    def __getitem__(self, index):
        file_path = self.files[index]
        state_dict = torch.load(file_path, map_location=self.device, weights_only=True)
        weights = []
        for weight in state_dict.values():
            weights.append(weight.flatten())
        return torch.hstack(weights)

    def get_state_dict(self, index):
        return torch.load(
            self.files[index], map_location=self.device, weights_only=True
        )

    def __len__(self):
        return len(self.files)


class DataHandler:
    def __init__(
        self,
        hparams: Dict[str, Any],
        data_folder: str,
        selectors: Union[DataSelector, List[DataSelector]],
        extract: bool = False,
    ):
        """
        Initialize DataHandler with flexible data selection.

        Args:
            hparams: Hyperparameters dictionary
            data_folder: Root folder containing the data
            selectors: One or more DataSelector objects specifying what data to use
            extract: Whether to extract additional information
        """
        self.hparams = hparams
        self.split_ratio = hparams.get("split_ratio", [80, 10, 10])
        self.data_path = data_folder

        # Convert single selector to list
        if isinstance(selectors, DataSelector):
            selectors = [selectors]

        # Get files based on selectors
        self.files = self._get_files_from_selectors(selectors)

        if len(self.files) == 0:
            raise ValueError(
                f"No files found matching the selection criteria in {data_folder}"
            )

        if "sample_limit" in hparams:
            if hparams["sample_limit"] > len(self.files):
                raise ValueError(
                    f"Sample limit {hparams['sample_limit']} is larger than available files ({len(self.files)})"
                )
            self.files = random.sample(self.files, hparams["sample_limit"])
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
                # For MNIST: match mnist_png_(train|test)_digit_id pattern
                if selector.dataset_type == DatasetType.MNIST:
                    # If class label is specified, match only that digit
                    if selector.class_label is not None:
                        if (
                            not f.startswith("mnist_png_")
                            or f"_{selector.class_label}_" not in f
                        ):
                            continue
                    else:
                        if not f.startswith("mnist_png_"):
                            continue

                    # If sample_id is specified, match that specific ID
                    if selector.sample_id is not None:
                        if not f.endswith(f"_{selector.sample_id}"):
                            continue

                # For CIFAR10: match cifar10_png_train_class_id pattern
                elif selector.dataset_type == DatasetType.CIFAR10:
                    if not f.startswith("cifar10_png_train"):
                        continue

                    # If class label is specified, match only that class
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
        self.train_dataset = self.val_dataset = self.test_dataset = IRNDataset(
            self.files, device=self.hparams["device"]
        )

    def _create_train_val_test_split(self):
        """Create train/val/test splits based on split ratio."""
        train_len = int(len(self.files) * self.split_ratio[0] / 100)
        val_len = int(len(self.files) * self.split_ratio[1] / 100)
        test_len = len(self.files) - train_len - val_len

        train_files, val_files, test_files = random_split(
            self.files, [train_len, val_len, test_len]
        )

        self.train_dataset = IRNDataset(train_files, device=self.hparams["device"])
        self.val_dataset = IRNDataset(val_files, device=self.hparams["device"])
        self.test_dataset = IRNDataset(test_files, device=self.hparams["device"])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
        )

    def get_state_dict(self, index):
        return self.train_dataset.get_state_dict(index)


# # Example usage
# if __name__ == "__main__":
#     # Example hyperparameters
#     hparams = {
#         "batch_size": 32,
#         "num_workers": 4,
#         "device": "cuda",
#         "split_ratio": [80, 10, 10],  # train/val/test split
#     }

#     # Example 1: Use all MNIST data
#     handler1 = DataHandler(
#         hparams,
#         "data/folder",
#         DataSelector(dataset_type=DatasetType.MNIST)
#     )

#     # Example 2: Use only MNIST class 1
#     handler2 = DataHandler(
#         hparams,
#         "data/folder",
#         DataSelector(dataset_type=DatasetType.MNIST, class_label=1)
#     )

#     # Example 3: Use MNIST class 1 with specific ID
#     handler3 = DataHandler(
#         hparams,
#         "data/folder",
#         DataSelector(dataset_type=DatasetType.MNIST, class_label=1, sample_id=100)
#     )

#     # Example 4: Use CIFAR10 airplanes
#     handler4 = DataHandler(
#         hparams,
#         "data/folder",
#         DataSelector(dataset_type=DatasetType.CIFAR10, class_label="airplane")
#     )

#     # Example 5: Use both MNIST and CIFAR10
#     handler5 = DataHandler(
#         hparams,
#         "data/folder",
#         [
#             DataSelector(dataset_type=DatasetType.MNIST),
#             DataSelector(dataset_type=DatasetType.CIFAR10)
#         ]
#     )

#     # Example 6: Use specific combinations
#     handler6 = DataHandler(
#         hparams,
#         "data/folder",
#         [
#             DataSelector(dataset_type=DatasetType.MNIST, class_label=1),
#             DataSelector(dataset_type=DatasetType.CIFAR10, class_label="airplane")
#         ]
#     )
