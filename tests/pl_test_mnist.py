#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /home/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch/tests/pl_test_mnist.py
Project: /home/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch/tests
Created Date: Thursday June 26th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday June 26th 2025 8:32:25 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F


# 1️⃣ 模型定义（LightningModule）
class LitClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        return self.layer2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# 2️⃣ 数据加载（DataLoader）
def get_dataloaders():
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader


# 3️⃣ 启动训练
if __name__ == "__main__":
    model = LitClassifier()
    train_loader = get_dataloaders()

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=2,  # ✅ 使用 2 张 GPU
        strategy="ddp",  # ✅ 使用 DDP 分布式策略
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader)
