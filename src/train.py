import warnings

import torch.nn.functional as F
from torch.optim import Adam, SGD

import pytorch_lightning as pl


warnings.filterwarnings("ignore", ".*TracerWarning.*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Set a lower value for log_every_n_steps.*")


class LightningModule(pl.LightningModule):
    def __init__(self, model, params):
        super().__init__()
        self.model = model
        self.params = params

        self.optimizer = {"Adam": Adam, "SGD": SGD}[params["optimizer"]]
        self.loss = F.mse_loss

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.params["learning_rate"])

    def _common_step(self, batch, batch_idx, stage):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        self.log(f"loss/{stage}", loss, on_step=False, on_epoch=True)
        return {"loss": loss, "y": y, "y_pred": y_pred}

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")
