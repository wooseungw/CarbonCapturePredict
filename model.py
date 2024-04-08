import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning as L
from models.segformer_simple import Segformer
import torchvision.models as models
def calculate_miou(preds, gt):
    intersection = torch.logical_and(preds, gt).sum()
    union = torch.logical_or(preds, gt).sum()
    miou = intersection / union
    return miou

class InitModel(L.LightningModule):
    def _cal_loss(self, batch, mode="train"):
        # 학습 단계에서의 로스 계산 및 로깅
        x , carbon, gt = batch
        preds = self(x)
        gt_loss = F.cross_entropy(preds, gt)
        carbon_loss = F.mse_loss(preds, carbon)
        miou = calculate_miou(preds, gt)
        mse = carbon_loss
        return gt_loss, carbon_loss, miou, mse
    def training_step(self, batch):

        gt_loss, carbon_loss, miou, mse = self._cal_loss(batch, mode="train")
        self.log("train_gt_loss", gt_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_MSE", mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_miou", miou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return gt_loss + carbon_loss

    def validation_step(self, batch, batch_idx):
        gt_loss, carbon_loss, miou, mse = self._cal_loss(batch, mode="val")
        self.log("Validation_gt_loss", gt_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation_MSE", mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation_miou", miou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return gt_loss + carbon_loss