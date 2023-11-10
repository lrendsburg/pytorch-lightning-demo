import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl


class ToyDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.train_dataset = self._generate_dataset(self.params["num_train"])
        self.val_dataset = self._generate_dataset(self.params["num_val"])

    def _generate_dataset(self, num_samples):
        X = torch.linspace(-1, 1, num_samples).view(-1, 1)
        Y = X * torch.sin(X * 10) + torch.normal(0, 0.05, X.shape)
        dataset = TensorDataset(X, Y)
        return dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.params["batch_size"], shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.params["batch_size"], shuffle=False
        )
