import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from models.dpt import DPTSegmentationWithCarbon
from dataset import CarbonDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn.functional as F
import wandb

def calculate_miou(preds, gt):
    intersection = torch.logical_and(preds, gt).sum()
    union = torch.logical_or(preds, gt).sum()
    miou = intersection / union
    return miou

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # PyTorch 1.9.0 이상에서 MPS 지원 확인
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = select_device()

model =DPTSegmentationWithCarbon().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
fp = 'Dataset/Training/image/SN10_Forest_IMAGE'
train_dataloader = DataLoader(CarbonDataset(transform=transform,mode='Train',folder_path=fp), batch_size=4, shuffle=True,num_workers=8)
val_dataloader = DataLoader(CarbonDataset(transform=transform,mode='Valid',folder_path=fp), batch_size=4, shuffle=True,num_workers=8)



def _cal_loss(self, batch):
    # 학습 단계에서의 로스 계산 및 로깅
    x , carbon, gt = batch
    gt_pred, carbon_pred = self(x)
    gt_loss = F.cross_entropy(gt_pred, gt)
    carbon_loss = F.mse_loss(carbon_pred, carbon)
    miou = calculate_miou(gt_pred, gt)
    
    return gt_loss, carbon_loss, miou
    
def training_step(self, batch):

    gt_loss, carbon_loss, miou, mse = self._cal_loss(batch)
    self.log("train_gt_loss", gt_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log("train_MSE", mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log("train_miou", miou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    loss = gt_loss + (carbon_loss*0.2)
    return loss

def validation_step(self, batch):
    gt_loss, carbon_loss, miou, mse = self._cal_loss(batch)
    self.log("Validation_gt_loss", gt_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log("Validation_MSE", mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log("Validation_miou", miou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    loss = gt_loss + (carbon_loss*0.2)
    return loss





