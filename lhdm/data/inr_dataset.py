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

    @classmethod
    def from_str(cls, value: Optional[str]) -> Optional["DatasetType"]:
        """Convert string to DatasetType, returning None if input is None."""
        if value is None:
            return None
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Invalid dataset type: {value}")


@dataclass
class DataSelector:
    """
    Flexible data selection criteria.

    Examples:
        # Select all data (both MNIST and CIFAR10)
        DataSelector(dataset_type=None)
        DataSelector()

        # Select all MNIST
        DataSelector(dataset_type=DatasetType.MNIST)

        # Select only MNIST class 1
        DataSelector(dataset_type=DatasetType.MNIST, class_label=1)

        # Select MNIST class 1 with specific ID
        DataSelector(dataset_type=DatasetType.MNIST, class_label=1, sample_id=100)
    """

    dataset_type: Optional[DatasetType]
    class_label: Optional[Union[int, str]] = None
    sample_id: Optional[int] = None


def create_selector_from_config(cfg: Dict[str, Any]) -> DataSelector:
    """Create a DataSelector from a configuration dictionary."""
    data_config = cfg.get("data", {})

    return DataSelector(
        dataset_type=DatasetType.from_str(data_config.get("dataset_type")),
        class_label=data_config.get("class_label"),
        sample_id=data_config.get("sample_id"),
    )


class INRDataset(Dataset):
    def __init__(self, files, device, not_flat=False):
        self.files = files
        self.device = device

        self.not_flat = not_flat

    def __getitem__(self, index):
        file_path = self.files[index]
        state_dict = torch.load(file_path, map_location=self.device, weights_only=True)

        if self.not_flat:
            return state_dict

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
        not_flat: bool = False,
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
        self.not_flat = not_flat

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

        self.train_dataset = INRDataset(
            train_files, device=self.hparams["device"], not_flat=self.not_flat
        )
        self.val_dataset = INRDataset(
            val_files, device=self.hparams["device"], not_flat=self.not_flat
        )
        self.test_dataset = INRDataset(
            test_files, device=self.hparams["device"], not_flat=self.not_flat
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
            persistent_workers=True,
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

#     # Example 1: Use all data (both MNIST and CIFAR10)
#     cfg1 = {}  # Empty config
#     handler1 = DataHandler(
#         hparams,
#         "data/folder",
#         create_selector_from_config(cfg1)
#     )

#     # Example 2: Use all MNIST data
#     cfg2 = {"data": {"dataset_type": "mnist"}}
#     handler2 = DataHandler(
#         hparams,
#         "data/folder",
#         create_selector_from_config(cfg2)
#     )

#     # Example 3: Use only MNIST class 1
#     cfg3 = {"data": {"dataset_type": "mnist", "class_label": 1}}
#     handler3 = DataHandler(
#         hparams,
#         "data/folder",
#         create_selector_from_config(cfg3)
#     )

#     # Example 4: Use MNIST class 1 with specific ID
#     cfg4 = {
#         "data": {
#             "dataset_type": "mnist",
#             "class_label": 1,
#             "sample_id": 100
#         }
#     }
#     handler4 = DataHandler(
#         hparams,
#         "data/folder",
#         create_selector_from_config(cfg4)
#     )

#     # Example 5: Use CIFAR10 airplanes
#     cfg5 = {
#         "data": {
#             "dataset_type": "cifar10",
#             "class_label": "airplane"
#         }
#     }
#     handler5 = DataHandler(
#         hparams,
#         "data/folder",
#         create_selector_from_config(cfg5)
#     )

#     # Example 6: Explicitly use None dataset type (matches both MNIST and CIFAR10)
#     cfg6 = {"data": {"dataset_type": None}}
#     handler6 = DataHandler(
#         hparams,
#         "data/folder",
#         create_selector_from_config(cfg6)
#     )

#     # Example 7: Multiple selectors
#     handler7 = DataHandler(
#         hparams,
#         "data/folder",
#         [
#             create_selector_from_config({"data": {"dataset_type": "mnist", "class_label": 1}}),
#             create_selector_from_config({"data": {"dataset_type": "cifar10", "class_label": "airplane"}})
#         ]
#     )
