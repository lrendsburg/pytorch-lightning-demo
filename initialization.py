import torch.nn as nn
import torch.nn.init as init

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.data_module import ToyDataModule
from src.train import LightningModule
from src.callbacks import (
    GradHistogramsCallback,
    ActivationsCallback,
)


class LinearSigmoid(nn.Module):
    def __init__(self, in_features, out_features, bias=True, skip_connection=False):
        super().__init__()
        self.skip_connection = skip_connection

        self.linear = nn.Linear(in_features, out_features)
        self.sigmoid = nn.Sigmoid()
        init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        if self.skip_connection:
            return x + self.sigmoid(self.linear(x))
        else:
            return self.sigmoid(self.linear(x))

    def forward(self, x):
        out = self.sigmoid(self.linear(x))
        if self.skip_connection:
            out += x
        return out


class SigmoidMLP(nn.Module):
    def __init__(self, layer_width, num_layers, skip_connections=False):
        super().__init__()

        layer_widths = [1] + [layer_width] * num_layers
        self.layers = []
        for i, (in_features, out_features) in enumerate(
            zip(layer_widths, layer_widths[1:])
        ):
            setattr(
                self,
                f"block{i+1}",
                LinearSigmoid(in_features, out_features, skip_connections),
            )
            self.layers.append(getattr(self, f"block{i+1}"))

        self.head = nn.Linear(layer_width, 1)
        self.layers.append(self.head)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def multiply_model_parameters(model, factor):
    for param in model.parameters():
        param.data *= factor


if __name__ == "__main__":
    # Hyperparameters
    params = {
        # Data module
        "num_train": 300,
        "num_val": 50,
        "batch_size": 64,
        # Training
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "max_epochs": 300,
    }

    # Setup
    data_module = ToyDataModule(params)
    model = SigmoidMLP(layer_width=300, num_layers=4, skip_connections=False)
    lightning_model = LightningModule(model, params)

    # Callbacks
    invoke_every = params["max_epochs"]  # Only call after first epoch
    callbacks = [
        GradHistogramsCallback(invoke_every),
        ActivationsCallback(invoke_every, activation_phase="before"),
        ActivationsCallback(invoke_every, activation_phase="after"),
    ]

    # Train
    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        callbacks=callbacks,
        logger=TensorBoardLogger("tb_logs", name="recipe"),
        enable_progress_bar=True,
    )
    trainer.fit(lightning_model, data_module)
