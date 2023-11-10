import logging

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from src.data_module import ToyDataModule
from src.model import NeuralNetwork
from src.train import LightningModule
from src.callbacks import MetricsCallback

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def objective(trial):
    # Search space
    params = {
        # Architecture
        "num_layers": 1,
        "layer_width": 50,
        # Data module
        "num_train": 300,
        "num_val": 50,
        "batch_size": 64,
        # Training
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-1, log=True),
        "max_epochs": 300,
    }

    # Setup
    data_module = ToyDataModule(params)
    model = NeuralNetwork(params)
    lightning_model = LightningModule(model, params)

    # Callbacks
    metrics_callback = MetricsCallback()
    callbacks = [
        metrics_callback,
        PyTorchLightningPruningCallback(trial, monitor="loss/val"),
        EarlyStopping(
            monitor="loss/val",
            min_delta=0.00,
            patience=15,
            mode="min",
        ),
    ]

    # Train
    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        callbacks=callbacks,
        logger=TensorBoardLogger("tb_logs", name="recipe"),
        enable_progress_bar=True,
    )
    trainer.fit(lightning_model, data_module)
    return trainer.callback_metrics["loss/val"].item()


if __name__ == "__main__":
    # Optuna study
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=30)

    best_params, best_value = study.best_params, study.best_value
    print(f"\n{best_value=} at {best_params=}")
