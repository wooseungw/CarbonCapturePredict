import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from .util import calculate_correlation, calculate_r2_score , batch_miou,select_device


class BaseModel(nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

class BaseModel_L(L.LightningModule):
    def __init__(self, num_classes):
        super(BaseModel_L, self).__init__()
        self.num_classes = num_classes
        #print("BASE num_classes",num_classes)
        weight = torch.tensor([0.0] + [1.0] * (self.num_classes - 1))
        self.seg_loss = nn.CrossEntropyLoss(weight=weight)
        self.reg_loss = nn.MSELoss()
        self._device = select_device()
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
        # 예측결과 가져오기
        gt_pred, carbon_pred = self(x)
        # print("GT 예측",gt_pred.shape,"GT:" ,gt.shape)
        # print("Carbon Pred",carbon_pred.shape,"Carbon:" ,carbon.shape)
        gt = gt.long()
        # 로스 계산
        gt_loss = self.seg_loss(gt_pred, gt)
        carbon_pred = carbon_pred.squeeze(1)
        carbon_loss = self.reg_loss(carbon_pred, carbon)
        # print("GT Loss",gt_loss)
        # print("carbon Loss",carbon_loss)
        # 평가지표 계산
        gt_pred = torch.argmax(gt_pred, dim=1)
        # print("평가지표 GT 예측",gt_pred.shape,"GT:" ,gt.shape)
        # print("평가지표 Carbon Pred",carbon_pred.shape,"Carbon:" ,carbon.shape)
        

        miou = batch_miou(gt_pred, gt, num_class=self.num_classes, device=self._device)
        corr = calculate_correlation(gt_pred,gt)
        r2 = calculate_r2_score(carbon_pred,carbon)
        # print("miou:",miou)
        # print("corr:",corr)
        # print("r2:",r2)

        return gt_loss, carbon_loss, miou , corr, r2
        
    def training_step(self, batch):

        gt_loss, carbon_loss, miou, corr,r2 = self._cal_loss(batch, mode="train")
        self.log("train_gt_loss", gt_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_MSE_loss", carbon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_miou", miou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_corr", corr,  on_epoch=True, prog_bar=True, logger=True)
        self.log("train_r2", r2,  on_epoch=True, prog_bar=True, logger=True)
        loss = gt_loss.float() + (carbon_loss.float()*0.005)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch):
        gt_loss, carbon_loss, miou, corr,r2 = self._cal_loss(batch, mode="val")
        self.log("Validation_gt_loss", gt_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validatrion_MSE_loss", carbon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation_miou", miou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation_corr", corr,  on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation_r2", r2,  on_epoch=True, prog_bar=True, logger=True)
        loss = gt_loss + (carbon_loss*0.2)
        return loss.float()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-6)
        return optimizer

