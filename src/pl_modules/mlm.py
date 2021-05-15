from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AdamW

from src.data.mlm import MLMDataset
from src.model.bert import BertMLM


class MLMModule(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = BertMLM(hparams['model_dir'])

    def forward(self, batch):
        input_ids = batch['input_ids']
        labels = batch['labels']
        return self.model(input_ids=input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs[0]
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs[0].detach().cpu().numpy()
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss = np.mean([out['loss'] for out in outputs])
        self.log('steps', self.global_step)
        self.log('val_loss', val_loss)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)


class MLMDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        dataset_dir = Path(self.cfg.dataset_dir)
        train_data = np.load(dataset_dir / 'train.npy', allow_pickle=True)
        valid_data = np.load(dataset_dir / 'val.npy', allow_pickle=True)
        self.train_dataset = MLMDataset(train_data)
        self.val_dataset = MLMDataset(valid_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg.train_loader)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.cfg.val_loader)
