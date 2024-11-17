import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import multiprocessing
import random

# Ensure 'spawn' start method is set globally for multiprocessing
multiprocessing.set_start_method("spawn", force=True)


class IRNDataset(Dataset):
    def __init__(self, files, device):
        self.files = files
        self.device = device

    def __getitem__(self, index):
        file_path = self.files[index]
        state_dict = torch.load(file_path, map_location=self.device)
        weights = []
        #print(state_dict.keys())
        for weight in state_dict.values():
            weights.append(weight.flatten())
        return torch.hstack(weights)
    
    def get_state_dict(self, index):
        return torch.load(self.files[index], map_location=self.device)

    def __len__(self):
        return len(self.files)


class DataHandler:
    def __init__(self, hparams, data_folder, fileprefix, extract=False):
        self.hparams = hparams
        self.split_ratio = hparams["split_ratio"]
        self.data_path = data_folder

        self.files = [
            os.path.join(
                os.path.join(os.path.join(data_folder, f), "checkpoints"),
                "model_final.pth",
            )
            for f in os.listdir(data_folder)
            if f.startswith(fileprefix)
        ]

        if "sample_limit" in hparams:
            self.files = random.sample(self.files, hparams["sample_limit"])
            self.train_dataset = self.val_dataset = self.test_dataset = IRNDataset(self.files, device=hparams["device"])
        else:
            # Split up into datasets according to split ratio
            train_dataset_length = int(len(self.files) * self.split_ratio[0] / 100)
            val_dataset_length = int(len(self.files) * self.split_ratio[1] / 100)
            test_dataset_length = int(
                len(self.files) - train_dataset_length - val_dataset_length
            )
            train_dataset, val_dataset, test_dataset = random_split(
                self.files, [train_dataset_length, val_dataset_length, test_dataset_length]
            )
            self.train_dataset = IRNDataset(train_dataset, device=hparams["device"])
            self.val_dataset = IRNDataset(val_dataset, device=hparams["device"])
            self.test_dataset = IRNDataset(test_dataset, device=hparams["device"])

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