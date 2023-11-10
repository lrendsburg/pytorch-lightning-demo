# Exploring PyTorch Lightning functionalities
This repository contains example usage of various PyTorch Lightning functionalities.
The following scripts demonstrate different aspects of PyTorch Lightning:
-  `main.py`: General usage of PyTorch Lightning
-  `sweep.py`: PyTorch Lightning with early stopping and hyperparameter tuning using optuna
-  `initialization.py`: Demonstration of using callbacks for investigating initialization behavior of the model

PyTorch Lightning reduces boilerplate code and promotes a clean separation of concerns. The main modules are 
1) **Data module**
2) **Model**
3) **Lightning module**
4) **Callbacks**
5) **Trainer**

## 1. Data module
The data-related logic is contained in a subclass of `pl.LightningDataModule`.
It is responsible for loading and preparing the data.
It  needs to implement functions such as `train_dataloader`, `val_dataloader`,  and `test_dataloader`, which return the data loader for the respective stage.
```python
class MyDataModule(pl.LightningDataModule):
  ...
  
  def train_dataloader(self):
    ...
    return train_loader
```
Alternatively, the data loaders can be passed directly to the `pl.LightningModule` instance or `trainer.fit`.

## 2. Model
The model itself can be specified as usual by an instance of `nn.Module`.
```python
class MyModel(nn.Module):
  ...

  def forward(self, x):
    ...

model = MyModel()
```
Alternatively, the forward pass can be directly specified in the `pl.LightningModule`, which is a subclass of `nn.Module`.

## 3. Lightning module
The training-related logic is contained in a subclass of `pl.LightningModule`. 
It only needs to provide methods for the inner loop of training in `training_step` and the optimizer in `configure_optimizers`. All other boilerplate logic and calls such as `.to(device)`, `model.train()`, `with torch.no_grad()`, ... are abstracted away by PyTorch Lightning.
```python
class MyLightningModule(pl.LightningModule):
  def __init__(self, model):
    self.model = model
    self.loss = ...
    ...

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_pred = self.model(x)
    loss = self.loss(y_pred, y)
    return loss

  def configure_optimizers(self):
    return Adam(self.parameters(), lr=1e-3)

```
Aside from the training_step, the lightning module can also allows hooks for various points during training such as
- `on_fit_start`
- `on_fit_end`
- `on_train_batch_start`
- `on_train_batch_end`
- `on_train_epoch_start`
- `on_train_epoch_end`
  
Similar hooks are also available for validation and testing.

Note that the `training_step` needs to return the loss, which is used for the optimization. Alternatively, it can return a dictionary that contains additional values for the training batch, e.g. 
```python
  def training_step(self, batch, batch_idx):
    x, y = batch
    y_pred = self.model(x)
    loss = self.loss(y_pred, y)
    return {"loss": loss, "y": y, "y_pred": y_pred}
```
This makes those values accessible to some hooks via their argument `outputs`, which can be useful for tasks such as computing additional metrics.


## 4. Callbacks
Hooks for various points during training can also be implemented in a subclass of `pl.Callback`. This keeps things more modular and can be used for logging, checkpointing, early stopping, etc.

```python
MetricsCallback(pl.Callback):
  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    # Accumulate metrics per batch

  def on_train_epoch_end(self, trainer, pl_module):
    # Log metrics
```
Note that every hook needs to accept a specific set of arguments; all hooks have access to the trainer instance `trainer` and the lightning module instance `pl_module`. Other hooks like `on_train_batch_end` also have access to the returned value from the `training_step` via the argument `outputs` and `batch` and `batch_idx` for the current batch.

## 5. Trainer
The trainer brings everything together. 

```python
data_module = MyDataModule()
model = MyModel()
lightning_module = MyLightningModule(model)

callbacks = [
  MetricsCallback(),
  EarlyStoppingCallback(),
  GradientHistogramCallback(),
  ...
]

trainer = pl.Trainer(max_epochs=10, callbacks=callbacks)
trainer.fit(lightning_module, data_module)
```

The trainer has various additional functionalities

**Accelerators.**
Accelerators can conveniently be specified by the trainer, eliminating the need to manually move the model and data to the device:
```python
trainer = Trainer(accelerator="gpu", devices=2)
```

**Logging.**
PyTorch Lightning has integration with various loggers such as TensorBoard, Comet, MLFlow, Neptune, Weights and Biases, ...
For example, tensorboard can be used as follows
```python
from pytorch_lightning.loggers import TensorBoardLogger
...
trainer = pl.Trainer(logger=TensorBoardLogger("logs/", name="my_model"))
```
The logger can then be accessed at the various hooks. There is the generic `self.log` in the lightning module, which can be used for logging arbitrary values. For example, to log the training loss with automatic epoch-wise aggregation, we can specify the `training_step` method in the lightning module as follows:
```python
def training_step(self, batch, batch_idx):
  x, y = batch
  y_pred = self.model(x)
  loss = self.loss(y_pred, y)
  self.log(f"loss/{stage}", loss, on_step=False, on_epoch=True)
  return {"loss": loss, "y": y, "y_pred": y_pred}
```
Alternatively, for more control, the specific logger can be accessed via the trainer. For example when using tensorboard, the tensorboard summary writer can be accessed via `trainer.logger.experiment`.

**Other utilities.**
The trainer provides several useful functionalities such as
- `trainer = pl.Trainer(fast_dev_run=True)`: Only runs a single batch to quickly check for errors
- `trainer = pl.Trainer(overfit_batches=1)`: Only trains on a single batch. If the model isn't able to overfit on a single batch, there is likely a bug in the code. See also [A recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)
- `trainer = pl.Trainer(profiler="simple")`: Lists the time spent in each part of execution to identify bottlenecks

**Warnings.**
The trainer automatically provides several helpful warnings for potential issues, for example when the validation loader uses `shuffle=True`. 
However in some cases, they spam the console and can be filtered with `warnings.filterwarnings("ignore", "...")`.

