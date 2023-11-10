import warnings
from typing import Any
from collections import defaultdict

import torch
import torch.nn as nn

from pytorch_lightning.callbacks import Callback
import torchmetrics

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)


def named_leaves(module, parent_name=""):
    """A deep version of module.named_children() that also returns leaves.

    This function is useful to access all layers if the model itself has subblocks."""
    for name, child in module.named_children():
        if list(child.children()):
            yield from named_leaves(child, parent_name + name + ".")
        else:
            yield parent_name + name, child


########################################################################################
########################## General purpose callbacks ###################################
########################################################################################


class MetricsCallback(Callback):
    def __init__(self):
        metrics = {
            "MAE": torchmetrics.MeanAbsoluteError,
            "R2": torchmetrics.R2Score,
        }
        self.train_metrics = {name: Metric() for name, Metric in metrics.items()}
        self.val_metrics = {name: Metric() for name, Metric in metrics.items()}

    def update_metrics(self, metrics, outputs, batch):
        y, y_pred = outputs["y"], outputs["y_pred"]
        for name, metric in metrics.items():
            metric.update(y_pred, y)

    def log_metrics(self, metrics, trainer, stage):
        for name, metric in metrics.items():
            epoch_metric = metric.compute()
            trainer.logger.experiment.add_scalar(
                f"{name}/{stage}", epoch_metric, trainer.current_epoch
            )
            metric.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.update_metrics(self.train_metrics, outputs, batch)

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_metrics(self.train_metrics, trainer, "train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.update_metrics(self.val_metrics, outputs, batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_metrics(self.val_metrics, trainer, "val")


class WeightHistogramsCallback(Callback):
    def __init__(self, invoke_every=1):
        super().__init__()
        self.invoke_every = invoke_every

    def log_weights(self, trainer, pl_module):
        for name, param in pl_module.model.named_parameters():
            trainer.logger.experiment.add_histogram(
                f"{name}/weights/epoch_{trainer.current_epoch}", param
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.invoke_every == 0:
            self.log_weights(trainer, pl_module)


class GradHistogramsCallback(Callback):
    def __init__(self, invoke_every=1):
        super().__init__()
        self.invoke_every = invoke_every
        self.grads = defaultdict(lambda: 0)

    def accumulate_grads(self, trainer, pl_module):
        for name, param in pl_module.model.named_parameters():
            if param.grad is not None:
                self.grads[name] += param.grad.data

    def log_grads(self, trainer, pl_module):
        for name, grads in self.grads.items():
            trainer.logger.experiment.add_histogram(
                f"{name}/gradients/epoch_{trainer.current_epoch}", grads
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.invoke_every == 0:
            self.accumulate_grads(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.invoke_every == 0:
            self.log_grads(trainer, pl_module)
        self.grads.clear()


class ActivationsCallback(Callback):
    def __init__(
        self,
        invoke_every=1,
        activations=(nn.ReLU, nn.Sigmoid),
        activation_phase="before",
    ):
        super().__init__()
        self.activations = activations
        self.invoke_every = invoke_every
        assert activation_phase in ["before", "after"]
        self.activation_phase = activation_phase

        self.values_dict = defaultdict(list)

    def make_hook_fn(self, name, trainer):
        def hook_fn(layer, input, output):
            if trainer.training and trainer.current_epoch % self.invoke_every == 0:
                values = input[0] if self.activation_phase == "before" else output
                values = values.detach().cpu().flatten()
                self.values_dict[name].extend(values)

        return hook_fn

    def log_values(self, trainer, pl_module):
        for name, values in self.values_dict.items():
            trainer.logger.experiment.add_histogram(
                f"{name}/{self.activation_phase}_activation/epoch_{trainer.current_epoch}",
                torch.tensor(values),
            )

    def on_fit_start(self, trainer, pl_module):
        for name, layer in named_leaves(pl_module.model):
            if isinstance(layer, self.activations):
                layer.register_forward_hook(self.make_hook_fn(name, trainer))

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_values(trainer, pl_module)
        self.values_dict.clear()


class ModelArchitectureCallback(Callback):
    def __init__(self):
        super().__init__()
        self.input_data = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.input_data = batch[0]

    def on_train_end(self, trainer, pl_module):
        trainer.logger.experiment.add_graph(pl_module.model, self.input_data)


########################################################################################
########################## Situational callbacks #######################################
########################################################################################


class LossMetricsCallback(Callback):
    def __init__(self):
        metrics = {
            "MAE": torchmetrics.MeanAbsoluteError,
            "R2": torchmetrics.R2Score,
            "loss": torchmetrics.MeanMetric,
        }
        self.train_metrics = {name: Metric() for name, Metric in metrics.items()}
        self.val_metrics = {name: Metric() for name, Metric in metrics.items()}

    def update_metrics(self, metrics, outputs, batch):
        y, y_pred = outputs["y"], outputs["y_pred"]
        for name, metric in metrics.items():
            if name == "loss":
                batch_size = len(batch[0])
                metric.update(outputs["loss"], weight=batch_size)
            else:
                metric.update(y_pred, y)

    def log_metrics(self, metrics, trainer, stage):
        for name, metric in metrics.items():
            epoch_metric = metric.compute()
            trainer.logger.experiment.add_scalar(
                f"{name}/{stage}", epoch_metric, trainer.current_epoch
            )
            metric.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.update_metrics(self.train_metrics, outputs, batch)

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_metrics(self.train_metrics, trainer, "train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.update_metrics(self.val_metrics, outputs, batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_metrics(self.val_metrics, trainer, "val")


class GridPredictionsCallback(Callback):
    def __init__(self, train_dataset, num_frames=10):
        super().__init__()
        self.train_dataset = train_dataset
        self.num_frames = num_frames

        self.grid = torch.linspace(-1, 1, 100).view(-1, 1)
        self.predictions = []

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            grid_pred = pl_module.model(self.grid)

        self.predictions.append(grid_pred)

    def on_train_end(self, trainer, pl_module):
        self.visualize()

    def visualize(self, filename="results/predictions.gif"):
        fig, ax = plt.subplots()
        X_train, Y_train = self.train_dataset[:]
        xlims = (torch.min(X_train), torch.max(X_train))
        ylims = (torch.min(Y_train), torch.max(Y_train))

        step_size = max(int(len(self.predictions) / self.num_frames), 1)
        predictions = self.predictions[::step_size]
        num_frames = len(predictions)

        def update(i):
            ax.clear()
            ax.scatter(*self.train_dataset[:], color="blue", alpha=0.3)
            for pred in predictions[:i]:
                ax.plot(self.grid, pred, color="red", alpha=0.3)
            ax.plot(self.grid, predictions[i], color="red")
            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)
            ax.set_title(f"Epoch {i * step_size}")

        ani = FuncAnimation(fig, update, frames=num_frames, blit=False)
        ani.save(filename, writer="imagemagick", fps=2)
        plt.close(fig)


class DeadNeuronsCallback(Callback):
    def __init__(self, invoke_every=1, visualize=True, log_dead_neurons_rate=True):
        super().__init__()
        self.invoke_every = invoke_every
        self.visualize = visualize
        self.log_dead_neurons_rate = log_dead_neurons_rate

        self.dead_neurons_dict = defaultdict(torch.Tensor)

    def make_hook_fn(self, name, trainer):
        def hook_fn(layer, input, output):
            if trainer.training and trainer.current_epoch % self.invoke_every == 0:
                mask = output == 0
                self.dead_neurons_dict[name] = torch.cat(
                    (self.dead_neurons_dict[name], mask), dim=0
                )

        return hook_fn

    def visualize_dead_neurons(self, trainer, pl_module):
        for name, mask in self.dead_neurons_dict.items():
            trainer.logger.experiment.add_image(
                f"{name}/dead_neurons/epoch_{trainer.current_epoch}",
                mask,
                dataformats="HW",
            )

    def log_num_dead_neurons(self, trainer, pl_module):
        mask = torch.cat([v.all(dim=0) for v in self.dead_neurons_dict.values()])
        dead_neuron_rate = mask.float().mean()
        trainer.logger.experiment.add_scalar(
            f"Fraction of dead neurons", dead_neuron_rate, trainer.current_epoch
        )

    def on_fit_start(self, trainer, pl_module):
        for name, layer in named_leaves(pl_module):
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(self.make_hook_fn(name, trainer))

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            trainer.current_epoch % self.invoke_every == 0
            and len(self.dead_neurons_dict) > 0
        ):
            if self.visualize:
                self.visualize_dead_neurons(trainer, pl_module)
            if self.log_dead_neurons_rate:
                self.log_num_dead_neurons(trainer, pl_module)
            self.dead_neurons_dict.clear()
