import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from .util import calculate_miou

class BaseModel(L.LightningModule):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters, strict=False)
    
    def _cal_loss(self, batch, mode="train"):
        # 학습 단계에서의 로스 계산 및 로깅
        x , carbon, gt = batch
        #carbon = carbon.to(dtype=torch.long)
        #gt = gt.to(dtype=torch.long)
        gt_pred, carbon_pred = self(x)
        gt = gt.long()
        gt_loss = F.cross_entropy(gt_pred, gt)
        carbon_pred = carbon_pred.squeeze(1)
        carbon_loss = F.mse_loss(carbon_pred, carbon)
        miou = calculate_miou(gt_pred, gt, 4)
        print("GT 예측",gt_pred.shape,"GT:" ,gt.shape)
        print("Carbon Pred",carbon_pred.shape,"Carbon:" ,carbon.shape)
        print("GT Loss",gt_loss)
        print("carbon Loss",carbon_loss)
        print("miou:",miou)
        return gt_loss, carbon_loss, miou
        
    def training_step(self, batch):

        gt_loss, carbon_loss, miou = self._cal_loss(batch, mode="train")
        self.log("train_gt_loss", gt_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_MSE_loss", carbon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_miou", miou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        loss = gt_loss.float() + (carbon_loss.float()*0.2)
        return loss
    @torch.no_grad()
    def validation_step(self, batch):
        gt_loss, carbon_loss, miou = self._cal_loss(batch, mode="val")
        self.log("Validation_gt_loss", gt_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validatrion_MSE_loss", carbon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation_miou", miou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        loss = gt_loss + (carbon_loss*0.2)
        return loss.float()
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-6)
        return optimizer