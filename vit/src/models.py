import os
import pandas as pd

import torch
from torch import nn
from vit_pytorch import ViT
import pytorch_lightning as pl

from torchmetrics import Accuracy, AUROC

from utils import plot_confusion_matrix


class VanillaDNN(pl.LightningModule):
    def __init__(self, settings):
        super(VanillaDNN, self).__init__()
        self.settings = settings
        self.save_hyperparameters()
        self.hidden_layers = nn.Sequential(
            nn.Linear(settings.VDNN_input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, settings.VDNN_num_classes),
        )

    def forward(self, X):
        return self.hidden_layers(X)


class VanillaViT(pl.LightningModule):
    def __init__(self, settings):
        super(VanillaViT, self).__init__()
        self.settings = settings
        self.save_hyperparameters()

        self.vit = ViT(
            image_size=tuple(settings.VVIT_image_size),
            patch_size=tuple(settings.VVIT_patch_size),
            num_classes=settings.VVIT_num_classes,
            dim=settings.VVIT_dim,
            depth=settings.VVIT_depth,
            heads=settings.VVIT_heads,
            mlp_dim=settings.VVIT_mlp_dim,
            pool=settings.VVIT_pool,
            channels=settings.VVIT_channels,
            dim_head=settings.VVIT_dim_head,
            dropout=settings.VVIT_dropout,
            emb_dropout=settings.VVIT_emb_dropout
        )

    def forward(self, X):
        return self.vit(X).squeeze()


class MultiClassClassifier(pl.LightningModule):
    def __init__(self,
                 settings):
        super(MultiClassClassifier, self).__init__()

        self.save_hyperparameters()
        self.settings = settings

        # PL attributes
        # Accuracy
        self.train_acc = Accuracy(task=settings.prediction_task,
                                  num_classes=settings.num_classes)
        self.valid_acc = Accuracy(task=settings.prediction_task,
                                  num_classes=settings.num_classes)
        self.test_acc = Accuracy(task=settings.prediction_task,
                                 num_classes=settings.num_classes)

        # AUROC
        self.train_auroc = AUROC(task=settings.prediction_task,
                                 num_classes=settings.num_classes)
        self.valid_auroc = AUROC(task=settings.prediction_task,
                                 num_classes=settings.num_classes)
        self.test_auroc = AUROC(task=settings.prediction_task,
                                num_classes=settings.num_classes)

        if settings.model == "VanillaDNN":
            self.transform = VanillaDNN(settings)
        elif settings.model == "VanillaViT":
            self.transform = VanillaViT(settings)
        else:
            raise ValueError(f"Model is invalid: {settings.model}")

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = settings.learning_rate
        self.softmax = nn.Softmax(dim=1)
        self.test_results = pd.DataFrame({
            "target": [],
            "pred": [],
            "dataloader_id": []
        })
        self.test_results_fp = os.path.join(
            settings.outdir, "test_results.csv")

    def forward(self, x):
        x = self.transform(x)
        return x

    def training_step(self, batch, batch_ids):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        pred_prob = self.softmax(y_hat)
        preds = torch.argmax(pred_prob, dim=1)
        self.train_acc.update(preds, y)
        self.train_auroc.update(pred_prob, y)
        self.log("train_loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        pred_prob = self.softmax(y_hat)
        preds = torch.argmax(pred_prob, dim=1)
        self.valid_acc.update(preds, y)
        self.valid_auroc.update(pred_prob, y)
        self.log("val_loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True)
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True)
        self.log("valid_auroc", self.valid_auroc.compute(), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        pred_prob = self.softmax(y_hat)
        preds = torch.argmax(pred_prob, dim=1)
        self.test_acc.update(preds, y)
        self.test_auroc.update(pred_prob, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        self.log("test_auroc", self.test_auroc.compute(), prog_bar=True)
        test_results = pd.DataFrame({
            'target': y.squeeze().cpu().numpy(),
            'pred': preds.squeeze().cpu().numpy(),
            'dataloader_id': [dataloader_idx for i in range(x.shape[0])]
        })
        self.test_results = pd.concat([self.test_results, test_results],
                                      axis=0, ignore_index=True)
        return loss

    def on_test_end(self):
        ref_df = self.test_results.copy(
        )[self.test_results['dataloader_id'] == 0].reset_index(drop=True)
        external_df = self.test_results.copy(
        )[self.test_results['dataloader_id'] != 0].reset_index(drop=True)
        plot_confusion_matrix(ref_df['target'], ref_df['pred'],
                              os.path.join(self.settings.outdir, "ref_cm.jpg"))
        if external_df.shape[0] > 0:
            plot_confusion_matrix(external_df['target'], external_df['pred'],
                                  os.path.join(self.settings.outdir, "external_cm.jpg"))

        self.test_results.to_csv(self.test_results_fp, index=False)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr)
        return optimizer
