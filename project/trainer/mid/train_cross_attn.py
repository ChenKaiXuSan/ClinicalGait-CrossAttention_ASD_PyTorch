"""
File: train_cross_attn.py
Project: project/trainer/mid
Created Date: 2026-07-24
Author: Kaixu Chen
-----
Comment:
 Trainer for CrossAttentionRes3DCNN (fuse_method=cross_atn).
 Classification-only loss, mirrors SEAttnTrainer so A1 fusion-method
 comparison stays apples-to-apples.

Have a good code time!
-----
Last Modified: Thursday July 24th 2026
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

"""

import logging

import torch
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

from project.models.cross_attn_res_3dcnn import CrossAttentionRes3DCNN

from project.utils.helper import save_helper

logger = logging.getLogger(__name__)


class CrossAttentionTrainer(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        self.img_size = hparams.data.img_size
        self.lr = getattr(hparams.loss, "lr", 1e-3)
        self.num_classes = int(hparams.model.model_class_num)

        # define model
        self.model = CrossAttentionRes3DCNN(hparams)

        # metrics（torchmetrics 多数支持 logits/probs，内部会做 argmax）
        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

        self.save_root = getattr(hparams.train, "log_path", "./logs")

    # ------------------- training / validation -------------------
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video: torch.Tensor = batch["video"]
        attn_map: torch.Tensor = batch["attn_map"]
        labels: torch.Tensor = batch["label"].long()
        B = video.size(0)

        logits = self.model(video, attn_map)

        probs = torch.softmax(logits, dim=1)
        loss_cls = F.cross_entropy(logits, labels)

        self.log("train/loss", loss_cls, on_step=True, on_epoch=True, batch_size=B)

        self.log_dict(
            {
                "train/video_acc": self._accuracy(probs, labels),
                "train/video_precision": self._precision(probs, labels),
                "train/video_recall": self._recall(probs, labels),
                "train/video_f1_score": self._f1_score(probs, labels),
            },
            on_step=True,
            on_epoch=True,
            batch_size=B,
        )

        logger.info(f"train loss: {loss_cls.item():.4f} ")
        return loss_cls

    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video: torch.Tensor = batch["video"]
        attn_map: torch.Tensor = batch["attn_map"]
        labels: torch.Tensor = batch["label"].long()
        B = video.size(0)

        logits = self.model(video, attn_map)

        probs = torch.softmax(logits, dim=1)
        loss_cls = F.cross_entropy(logits, labels)

        self.log("val/loss", loss_cls, on_step=False, on_epoch=True, batch_size=B)

        self.log_dict(
            {
                "val/video_acc": self._accuracy(probs, labels),
                "val/video_precision": self._precision(probs, labels),
                "val/video_recall": self._recall(probs, labels),
                "val/video_f1_score": self._f1_score(probs, labels),
            },
            on_step=False,
            on_epoch=True,
            batch_size=B,
        )

        logger.info(f"val loss: {loss_cls.item():.4f} ")
        return {"val_loss": loss_cls}

    # ------------------- testing -------------------
    def on_test_start(self) -> None:
        self.test_pred_list: list[torch.Tensor] = []
        self.test_label_list: list[torch.Tensor] = []
        logger.info("test start")

    def on_test_end(self) -> None:
        logger.info("test end")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video: torch.Tensor = batch["video"]
        attn_map: torch.Tensor = batch["attn_map"]
        labels: torch.Tensor = batch["label"].long()
        B = video.size(0)

        logits = self.model(video, attn_map)
        probs = torch.softmax(logits, dim=1)

        loss = F.cross_entropy(logits, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=B)

        self.log_dict(
            {
                "test/video_acc": self._accuracy(probs, labels),
                "test/video_precision": self._precision(probs, labels),
                "test/video_recall": self._recall(probs, labels),
                "test/video_f1_score": self._f1_score(probs, labels),
            },
            on_step=False,
            on_epoch=True,
            batch_size=B,
        )

        self.test_pred_list.append(probs.detach().cpu())
        self.test_label_list.append(labels.detach().cpu())

        return probs, logits

    def on_test_epoch_end(self) -> None:
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=(
                getattr(self.logger, "root_dir", "fold").split("/")[-1]
                if self.logger
                else "fold"
            ),
            save_path=self.save_root,
            num_class=self.num_classes,
        )
        logger.info("test epoch end")

    # ------------------- optimizer/scheduler -------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        tmax = getattr(self.trainer, "estimated_stepping_batches", None)
        if not isinstance(tmax, int) or tmax <= 0:
            tmax = 1000

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train/loss"},
        }
