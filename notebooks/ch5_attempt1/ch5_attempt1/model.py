"""
General regression module, implemented as a Pytorch Lightning module
"""

import logging
import torch
import pandas as pd
import numpy as np
from serotiny.networks import MLP

logger = logging.getLogger("lightning")

from serotiny.models.base_model import BaseModel

class Classifier(BaseModel):
    def __init__(self, x_label, y_label, network, loss, **kwargs):
        super().__init__()
        self.network = network
        self.loss = loss
        self.x_label = x_label
        self.y_label = y_label
        self.network 


    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def parse_batch(self, batch):
        return batch[self.hparams.x_label], batch[self.hparams.y_label]

    #def predict_step(self, batch, batch_idx):
     #   x, y = self.parse_batch(batch)
     #  return self.network(x), y

    def _step(self, stage, batch, batch_ids, logger):
        x, y = self.parse_batch(batch)

        yhat = self.network(x)

        loss = self.loss(yhat.squeeze(),y.squeeze())
        if stage != "predict":
            self.log(f"{stage}_loss", loss.detach(), logger=logger)

        results = {
            "loss": loss,
            "yhat": yhat.detach().squeeze(),
            "ytrue": y.detach().squeeze(),
            "split": batch["split"],
            "cell_id": batch["cell_id"]
        }

        return results
