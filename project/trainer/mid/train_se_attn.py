"""
File: train.py
Project: project
Created Date: 2023-10-19 02:29:47
Author: chenkaixu
-----
Comment:
 This file is the train/val/test process for the project.


Have a good code time!
-----
Last Modified: Thursday May 1st 2025 8:34:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

"""

from typing import Any, List, Optional, Union
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

from project.models.pose_fusion_res_3dcnn import PoseFusionRes3DCNN
from project.models.se_attn_res_3dcnn import SEFusion3DCNN

from project.utils.helper import save_helper

logger = logging.getLogger(__name__)


class PoseAttnTrainer(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()  # 先保存，方便 ckpt/repro

        self.img_size = hparams.data.img_size
        self.lr = float(hparams.optimizer.lr)
        self.num_classes = int(hparams.model.model_class_num)

        # define model
        self.model = PoseFusionRes3DCNN(hparams)

        # metrics（torchmetrics 多数支持 logits/probs，内部会做 argmax）
        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

        # loss 权重（可在 YAML 覆盖）
        self.lambda_list = list(
            getattr(self.hparams, "lambda_list", [0.25, 0.5, 0.75, 1.0])
        )
        self.w_bg = float(getattr(self.hparams, "w_bg", 0.2))
        self.w_temp = float(getattr(self.hparams, "w_temp", 0.05))

    # ------------------- small helpers -------------------
    @staticmethod
    def _resize_3d(x: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)

    @staticmethod
    def _bce_dice_loss(
        logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum(dim=(1, 2, 3, 4))
        denom = p.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4)) + eps
        dice = 1.0 - (2.0 * inter + eps) / denom
        return bce + dice.mean()

    @staticmethod
    def _temporal_tv_l1(prob: torch.Tensor) -> torch.Tensor:
        # prob: (B,C,T,H,W) in [0,1]
        if prob.size(2) <= 1:
            return prob.new_zeros(())
        return (prob[:, :, 1:] - prob[:, :, :-1]).abs().mean()

    def _compute_attn_losses(
        self,
        side_preds: list[torch.Tensor],  # 每层侧头 logits: (B,J,Ti,Hi,Wi)
        doctor_hm: torch.Tensor,  # (B,J,T,H,W) in [0,1]
        visible_mask: torch.Tensor,  # (B,J,T,H,W) or None
    ) -> dict[str, torch.Tensor]:
        if len(side_preds) == 0:
            z = doctor_hm.new_zeros(())
            return {"attn": z, "bg": z, "tmp": z}

        loss_attn_total = doctor_hm.new_zeros(())
        loss_bg_total = doctor_hm.new_zeros(())
        loss_tmp_total = doctor_hm.new_zeros(())

        prev_up = None
        for i, Pi in enumerate(side_preds):
            Ti, Hi, Wi = Pi.shape[2:]
            Ai = self._resize_3d(doctor_hm, (Ti, Hi, Wi))

            if visible_mask is not None:
                Mi = self._resize_3d(visible_mask, (Ti, Hi, Wi))
                attn_loss = self._bce_dice_loss(Pi * Mi, Ai * Mi)
            else:
                attn_loss = self._bce_dice_loss(Pi, Ai)

            # 背景抑制：并集 → 背景
            A_union = Ai.max(dim=1, keepdim=True).values
            A_bg = (1.0 - A_union).clamp(0, 1)
            P_max = Pi.max(dim=1, keepdim=True).values
            bg_loss = F.binary_cross_entropy_with_logits(
                P_max, torch.zeros_like(P_max), weight=A_bg
            )

            # 时间平滑（在 prob 上）
            P_sig = torch.sigmoid(Pi)
            tmp_loss = self._temporal_tv_l1(P_sig)

            lam = (
                self.lambda_list[i]
                if i < len(self.lambda_list)
                else self.lambda_list[-1]
            )
            loss_attn_total = loss_attn_total + lam * attn_loss
            loss_bg_total = loss_bg_total + self.w_bg * bg_loss
            loss_tmp_total = loss_tmp_total + self.w_temp * tmp_loss

            prev_up = Pi  # 若后续需要层间一致性，可在此加入 KL 蒸馏

        return {"attn": loss_attn_total, "bg": loss_bg_total, "tmp": loss_tmp_total}

    # ------------------- training / validation -------------------
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video: torch.Tensor = batch["video"]
        attn_map: torch.Tensor = batch["attn_map"]
        labels: torch.Tensor = batch["label"].long()
        B = video.size(0)
        visible_mask: torch.Tensor | None = batch.get("attn_mask", None)

        out = self.model(video, attn_map, return_aux=True)
        logits, aux = out if isinstance(out, tuple) else (out, {"side_preds": []})

        probs = torch.softmax(logits, dim=1)
        loss_cls = F.cross_entropy(logits, labels)

        attn_losses = self._compute_attn_losses(
            aux.get("side_preds", []), doctor_hm=attn_map, visible_mask=visible_mask
        )
        loss_total = (
            loss_cls + attn_losses["attn"] + attn_losses["bg"] + attn_losses["tmp"]
        )

        # logging
        self.log("train/loss", loss_total, on_step=True, on_epoch=True, batch_size=B)
        self.log("train/loss_cls", loss_cls, on_step=True, on_epoch=True, batch_size=B)
        if len(aux.get("side_preds", [])) > 0:
            self.log(
                "train/loss_attn",
                attn_losses["attn"],
                on_step=True,
                on_epoch=True,
                batch_size=B,
            )
            self.log(
                "train/loss_bg",
                attn_losses["bg"],
                on_step=True,
                on_epoch=True,
                batch_size=B,
            )
            self.log(
                "train/loss_tmp",
                attn_losses["tmp"],
                on_step=True,
                on_epoch=True,
                batch_size=B,
            )

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

        logger.info(
            f"train loss: {loss_total.item():.4f} "
            f"(cls {loss_cls.item():.4f} | attn {attn_losses['attn'].item():.4f} | "
            f"bg {attn_losses['bg'].item():.4f} | tmp {attn_losses['tmp'].item():.4f})"
        )
        return loss_total

    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video: torch.Tensor = batch["video"]
        attn_map: torch.Tensor = batch["attn_map"]
        labels: torch.Tensor = batch["label"].long()
        B = video.size(0)
        visible_mask: torch.Tensor | None = batch.get("attn_mask", None)

        out = self.model(video, attn_map, return_aux=True)
        logits, aux = out if isinstance(out, tuple) else (out, {"side_preds": []})

        probs = torch.softmax(logits, dim=1)
        loss_cls = F.cross_entropy(logits, labels)

        attn_losses = self._compute_attn_losses(
            aux.get("side_preds", []), doctor_hm=attn_map, visible_mask=visible_mask
        )
        loss_total = (
            loss_cls + attn_losses["attn"] + attn_losses["bg"] + attn_losses["tmp"]
        )

        # 建议验证只 on_epoch 记录，减少噪声
        self.log("val/loss", loss_total, on_step=False, on_epoch=True, batch_size=B)
        self.log("val/loss_cls", loss_cls, on_step=False, on_epoch=True, batch_size=B)
        if len(aux.get("side_preds", [])) > 0:
            self.log(
                "val/loss_attn",
                attn_losses["attn"],
                on_step=False,
                on_epoch=True,
                batch_size=B,
            )
            self.log(
                "val/loss_bg",
                attn_losses["bg"],
                on_step=False,
                on_epoch=True,
                batch_size=B,
            )
            self.log(
                "val/loss_tmp",
                attn_losses["tmp"],
                on_step=False,
                on_epoch=True,
                batch_size=B,
            )

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

        logger.info(
            f"val loss: {loss_total.item():.4f} "
            f"(cls {loss_cls.item():.4f} | attn {attn_losses['attn'].item():.4f} | "
            f"bg {attn_losses['bg'].item():.4f} | tmp {attn_losses['tmp'].item():.4f})"
        )
        return {"val_loss": loss_total}

    # ------------------- testing -------------------
    def on_test_start(self) -> None:
        self.test_pred_list: list[torch.Tensor] = []
        self.test_label_list: list[torch.Tensor] = []
        logger.info("test start")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video: torch.Tensor = batch["video"]
        attn_map: torch.Tensor = batch["attn_map"]
        labels: torch.Tensor = batch["label"].long()
        B = video.size(0)

        out = self.model(video, attn_map, return_aux=False)
        logits = out if isinstance(out, torch.Tensor) else out[0]
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
        return {"probs": probs, "logits": logits}

    def on_test_epoch_end(self) -> None:
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=(
                getattr(self.logger, "root_dir", "fold").split("/")[-1]
                if self.logger
                else "fold"
            ),
            save_path=getattr(self.logger, "save_dir", "."),
            num_class=self.num_classes,
        )
        logger.info("test epoch end")

    # ------------------- optimizer/scheduler -------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Lightning 里 estimated_stepping_batches 可能在早期不可用，做个稳健 fallback
        tmax = getattr(self.trainer, "estimated_stepping_batches", None)
        if not isinstance(tmax, int) or tmax <= 0:
            tmax = 1000  # 安全兜底，后续也可换 OneCycleLR

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train/loss"},
        }
