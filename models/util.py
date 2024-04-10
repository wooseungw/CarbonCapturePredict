import torch
from cmath import nan
import numpy as np
import torch
import numpy as np
from sklearn.metrics import r2_score


def calculate_correlation(preds, labels):
    # 텐서 평탄화
    preds_flat = preds.view(preds.size(0), -1)
    labels_flat = labels.view(labels.size(0), -1)
    
    # 배치별 상관관계 계산
    correlation_scores = []
    for i in range(preds_flat.size(0)):
        corr = np.corrcoef(preds_flat[i].detach().numpy(), labels_flat[i].detach().numpy())[0, 1]
        correlation_scores.append(corr)
    
    # 평균 상관관계 반환
    return np.mean(correlation_scores)

def calculate_r2_score(tensor_true, tensor_pred):
    # 텐서 평탄화
    tensor_true_flat = tensor_true.view(tensor_true.size(0), -1).detach().numpy()
    tensor_pred_flat = tensor_pred.view(tensor_pred.size(0), -1).detach().numpy()
    
    # 배치별 R² 점수 계산
    r2_scores = [r2_score(true, pred) for true, pred in zip(tensor_true_flat, tensor_pred_flat)]
    
    # 평균 R² 점수 반환
    return np.mean(r2_scores)

def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = torch.bincount(
        n_class * label_true[mask] +
        label_pred[mask], minlength=n_class ** 2
    ).reshape(n_class, n_class)
    return hist

def compute_miou(hist):
    with torch.no_grad():
        iou = torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist))
        miou = torch.nanmean(iou)
    return miou

def batch_miou(label_preds, label_trues, num_class):
    """Calculate mIoU for a batch of predictions and labels"""
    hist = torch.zeros((num_class, num_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += fast_hist(lt.flatten(), lp.flatten(), num_class)
    return compute_miou(hist)

if __name__ == "__main__":
  # 예제 데이터
  n_class = 3  # 클래스 수
  batch_size = 10
  height, width = 224, 224  # 예제 이미지 크기

  # 임의의 실제 라벨과 예측 라벨 생성
  label_trues = torch.randint(0, n_class, (batch_size, height, width))
  label_preds = torch.randint(0, n_class, (batch_size, height, width))

  # 배치 mIoU 계산
  miou = batch_miou(label_preds,label_trues, n_class)
  print(f"Batch mIoU: {miou.item()}")


  # 상관관계 계산
  correlation = calculate_correlation(label_preds, label_trues)
  print(f"Correlation: {correlation}")


  pred_carbon = torch.rand((batch_size,1, height, width))
  label_carbon = torch.rand((batch_size,1, height, width))
  # R² 점수 계산
  r2_score = calculate_r2_score(pred_carbon, label_carbon)
  print(f"R² Score: {r2_score}")