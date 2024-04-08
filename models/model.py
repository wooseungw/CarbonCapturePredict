import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import timm

def calculate_miou(preds, gt):
    intersection = torch.logical_and(preds, gt).sum()
    union = torch.logical_or(preds, gt).sum()
    miou = intersection / union
    return miou

def modify_first_conv_layer(model, new_in_channels=4):
    print(model)
    # 첫 번째 컨볼루션 레이어를 가져옵니다.
    first_conv_layer = model.patch_embed.proj
    old_in_channels = first_conv_layer.in_channels

    # 새로운 컨볼루션 레이어를 생성합니다.
    # 기존 레이어에서 bias가 설정되어 있는지 확인하고, 그에 따라 새 레이어의 bias 매개변수를 설정합니다.
    new_first_conv_layer = nn.Conv2d(
        in_channels=new_in_channels, 
        out_channels=first_conv_layer.out_channels, 
        kernel_size=first_conv_layer.kernel_size, 
        stride=first_conv_layer.stride, 
        padding=first_conv_layer.padding, 
        bias=(first_conv_layer.bias is not None)  # bias가 None이 아니면 True, None이면 False
    )

    # 기존 가중치를 새 레이어에 복사합니다.
    with torch.no_grad():
        new_first_conv_layer.weight[:, :old_in_channels] = first_conv_layer.weight
        # 새 채널에 대한 가중치는 기존 채널의 평균으로 초기화합니다.
        new_first_conv_layer.weight[:, old_in_channels:] = first_conv_layer.weight[:, :1].mean(dim=1, keepdim=True)
        if first_conv_layer.bias is not None:
            new_first_conv_layer.bias = first_conv_layer.bias

    # 모델의 첫 번째 컨볼루션 레이어를 새로운 레이어로 교체합니다.
    model.patch_embed.proj = new_first_conv_layer

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

    def validation_step(self, batch):
        gt_loss, carbon_loss, miou, mse = self._cal_loss(batch, mode="val")
        self.log("Validation_gt_loss", gt_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation_MSE", mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation_miou", miou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return gt_loss + carbon_loss
    

class DPT4d(InitModel):
    def __init__(self,backbone = 'vit_base_resnet50d_224' ):
        super(DPT4d, self).__init__()
        self.backbone = timm.create_model(backbone,in_chans=4,num_classes=0)
        

        
    def forward(self, x):
        output = self.backbone(x)
        return output
    
if __name__=='__main__':
    model = DPT4d()
    x = torch.randn(1, 3, 224, 224)
    print(model.forward(torch.randn(1, 4, 224, 224)).shape)
    #print(timm.list_models())