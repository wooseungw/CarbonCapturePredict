from cmath import nan
import torch
from torch import nn
import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from .evaluate import corr, r_square, corr_wZero, r_square_wZero
from .util import batch_miou

class DANNLoss(nn.Module):
    def __init__(self, cls_lambda=1, reg_lambda=0.0005, num_classes = 4) -> None:
        super().__init__()
        self.carbon_loss = CarbonLoss(cls_lambda = cls_lambda, reg_lambda = reg_lambda,num_classes =num_classes)
        self.domain_loss = nn.BCELoss()

class CarbonLoss(nn.Module):
    def __init__(self, cls_lambda=1, reg_lambda=0.0005, num_classes = 4):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.ce = nn.CrossEntropyLoss(torch.tensor([0.] + [1.] * (num_classes-1), dtype=torch.float))
        self.cls_lambda = cls_lambda
        self.reg_lambda = reg_lambda
        self.num_classes = num_classes
    def forward(self, input_cls, target_cls, input_reg , target_reg):
        
        target_cls = target_cls.long()
        cls_loss = self.ce(input_cls, target_cls)
        _, input_cls = torch.max(input_cls, dim=1)
        miou = batch_miou(input_cls, target_cls, self.num_classes, torch.device("cpu"))
        reg_loss = self.mse(
            torch.flatten(input_reg, end_dim=-2),
            torch.flatten(target_reg, end_dim=-2)
        )
        input_reg = input_reg.squeeze()
        target_reg = target_reg.squeeze()

        corr_sum=0
        r_sum =0
        idx = 1e-9
        for i in range(len(input_reg)):
            ir = input_reg[i,...].cpu().detach().numpy()
            tr = target_reg[i,...].cpu().detach().numpy()
            if np.count_nonzero(tr) == 0:
                continue          
            corr_res = corr_wZero(ir,tr)
            r_res = r_square_wZero(ir,tr)
            if np.isnan(corr_res) or np.isnan(r_res):           
                continue
            corr_sum = corr_sum + corr_res
            r_sum = r_sum + r_res
            idx = idx +1
        acc_c = corr_sum/idx
        acc_r = r_sum/idx

        total_loss = self.cls_lambda * cls_loss + self.reg_lambda * reg_loss

        return total_loss, cls_loss, reg_loss, acc_c, acc_r, miou