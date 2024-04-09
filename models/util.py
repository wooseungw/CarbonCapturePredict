import torch
from cmath import nan
import numpy as np

def corr(A,B):
  ################ with zero
    # a = A-A.mean()
    # b = B-B.mean()
    # return (a*b).sum() / (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum()))

  ################ without zero
  A = np.where(B!=0,A,np.nan)
  B = np.where(B!=0,B,np.nan)
  a = A-np.nanmean(A)
  b = B-np.nanmean(B)
  if (np.sqrt(np.nansum(a**2)) * np.sqrt(np.nansum(b**2))) == 0:
    return nan
  else: 
    return np.nansum(a*b) / (np.sqrt(np.nansum(a**2)) * np.sqrt(np.nansum(b**2)))


def corr_wZero(preds,target):
    preds = preds.float()
    target = target.float()
    a = preds - preds.mean()
    b = target - target.mean()
    denom = torch.sqrt((a**2).sum()) * torch.sqrt((b**2).sum())
    if denom == 0:
        return float('nan')
    else:
        return (a*b).sum() / denom
 
def corr_wCla(A,B,C):  
  A = np.where(C!=255,A,np.nan)
  B = np.where(C!=255,B,np.nan)  
  # A = np.where(B!=0,A,np.nan)
  # B = np.where(B!=0,B,np.nan)
  # B_nonnan_cnt = np.count_nonzero(~np.isnan(B))
  # B_nz_cnt = np.count_nonzero(B)
  a = A-np.nanmean(A)
  b = B-np.nanmean(B)
  if (np.sqrt(np.nansum(a**2)) * np.sqrt(np.nansum(b**2))) == 0:
    return nan#, B_nonnan_cnt, B_nz_cnt
  else: 
    return np.nansum(a*b) / (np.sqrt(np.nansum(a**2)) * np.sqrt(np.nansum(b**2)))#, B_nonnan_cnt, B_nz_cnt
 
########################################################################################################################
  
def r_square(A,B):  
  A = np.where(B!=0,A,np.nan)
  B = np.where(B!=0,B,np.nan)
  if np.nansum((B-np.nanmean(B))**2) == 0:
    return nan
  else:
    return 1.0 - ( np.nansum(((B-A)**2))  / np.nansum((B-np.nanmean(B))**2) )

def r_square_wZero(preds, target):
    preds = preds.float()
    target = target.float()
    if torch.nansum((target - torch.nanmean(target))**2) == 0:
        return float('nan')
    else:
        return 1.0 - ((target - preds)**2).sum() / ((target - target.mean())**2).sum()


def r_square_wCla(A,B,C):  
  A = np.where(C!=255,A,np.nan)
  B = np.where(C!=255,B,np.nan)
  # A = np.where(B!=0,A,np.nan)
  # B = np.where(B!=0,B,np.nan)
  if np.nansum((B-np.nanmean(B))**2) == 0:
    return nan
  else:
    return 1.0 - ( np.nansum(((B-A)**2))  / np.nansum((B-np.nanmean(B))**2) )  

def calculate_miou(preds, gt, num_classes):
    miou_sum = 0.0
    # preds에서 최대값의 인덱스를 얻습니다. 이는 각 픽셀에 대한 예측된 클래스입니다.
    _, preds = torch.max(preds, 1) 
    
    for class_index in range(num_classes):
        pred_mask = (preds == class_index)  # 예측된 클래스 마스크
        gt_mask = (gt == class_index)  # 실제 클래스 마스크
        intersection = torch.logical_and(pred_mask, gt_mask).sum()
        union = torch.logical_or(pred_mask, gt_mask).sum()
        # 분모가 0인 경우를 처리하기 위해 작은 epsilon 값을 추가합니다.
        iou = intersection / (union + 1e-7)
        miou_sum += iou

    miou = miou_sum / num_classes
    return miou

if __name__ == "__main__":
    # Create random tensors
    num_classes= 4
    batch = 10
    ch = num_classes
    w = 10
    h = 10
    preds = torch.randint(0, num_classes, (batch, ch, w, h))
    _,gt = torch.max(preds, 1) 
    
    print(gt.shape)
    print(preds.shape)
    # Calculate mIoU
    miou = calculate_miou(preds, gt, num_classes)
    print(miou)