import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.data_module import ToyDataModule
from src.model import NeuralNetwork
from src.train import LightningModule
from src.callbacks import (
    MetricsCallback,
    GridPredictionsCallback,
    ModelArchitectureCallback,
    WeightHistogramsCallback,
    GradHistogramsCallback,
    ActivationsCallback,
    DeadNeuronsCallback,
)


if __name__ == "__main__":
    # Hyperparameters
    params = {
        # Architecture
        "num_layers": 1,
        "layer_width": 50,
        # Data module
        "num_train": 300,
        "num_val": 50,
        "batch_size": 64,
        # Training
        "optimizer": "Adam",
        "learning_rate": 5e-2,
        "max_epochs": 300,
    }

    # Setup
    data_module = ToyDataModule(params)
    model = NeuralNetwork(params)
    lightning_model = LightningModule(model, params)

    # Callbacks
    invoke_every = max(int(params["max_epochs"] / 5), 1)
    callbacks = [
        MetricsCallback(),
        ModelArchitectureCallback(),
        WeightHistogramsCallback(invoke_every),
        GradHistogramsCallback(invoke_every),
        ActivationsCallback(invoke_every, activation_phase="before"),
        ActivationsCallback(invoke_every, activation_phase="after"),
        DeadNeuronsCallback(invoke_every, visualize=True, log_dead_neurons_rate=True),
        GridPredictionsCallback(data_module.train_dataset, num_frames=20),
    ]

    # Train
    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        callbacks=callbacks,
        logger=TensorBoardLogger("tb_logs", name="recipe"),
        enable_progress_bar=True,
        # overfit_batches=1, # only train on a small part of the training set
        # fast_dev_run=True,  # fast debugging
        # profiler="simple",  # Profile the runtime
    )
    trainer.fit(lightning_model, data_module)
