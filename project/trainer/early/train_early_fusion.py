"""
File: train_early_fusion.py
Project: project/trainer/early
Created Date: 2024-05-13 (rewritten 2026-07-24)
Author: Kaixu Chen
-----
Comment:
 Trainer for early-fusion Res3DCNN (fuse_method in {add, mul, concat, avg}).
 Res3DCNN.forward(video, attn_map) performs the fusion internally at the input,
 so this is a plain single-model classification trainer (cls loss only),
 mirroring SEAttnTrainer / CrossAttentionTrainer.

 NOTE: replaces a stale two-stream (stance/swing) trainer that indexed
 batch["video"][..., 0/1] and used self.stance_cnn/self.swing_cnn — a leftover
 that never matched the current (B,C,T,H,W) + attn_map data format.

Have a good code time!
-----
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

from project.models.make_model import select_model

from project.utils.helper import save_helper

logger = logging.getLogger(__name__)


class EarlyFusion3DCNNTrainer(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        self.img_size = hparams.data.img_size
        self.lr = getattr(hparams.loss, "lr", 1e-3)  # lr lives under loss, not optimizer
        self.num_classes = int(hparams.model.model_class_num)

        # Res3DCNN with fuse_method=add/mul/concat/avg (fusion done inside forward)
        self.model = select_model(hparams)

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
