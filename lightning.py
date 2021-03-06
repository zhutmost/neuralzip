import pytorch_lightning as pl
import torch as t
from omegaconf import DictConfig

from apputil import load_obj


class LitModuleWrapper(pl.LightningModule):
    def __init__(self, model: t.nn.Module, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg

        self.criterion = t.nn.CrossEntropyLoss()

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy(dist_sync_on_step=True)

        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        self.train_acc(outputs, targets)
        metrics = {'acc': self.train_acc}
        self.log_dict(metrics, prog_bar=True)
        return {'loss': loss, **metrics}

    def eval_common_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        self.val_acc(outputs, targets)
        metrics = {'val_loss': loss, 'val_acc': self.val_acc}
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self.eval_common_step(batch, batch_idx)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        metrics = self.eval_common_step(batch, batch_idx)
        metrics = {'test_acc': metrics['val_acc'], 'test_loss': metrics['val_loss']}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer_fn = load_obj(self.cfg.optimizer.class_name, 'torch.optim')
        lr_scheduler_fn = load_obj(self.cfg.lr_scheduler.class_name, 'torch.optim.lr_scheduler')
        optimizer = optimizer_fn(self.parameters(), **self.cfg.optimizer.params)
        lr_scheduler = lr_scheduler_fn(optimizer, **self.cfg.lr_scheduler.params)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
