import torch
from cmath import nan
import numpy as np
import torch
import numpy as np
from sklearn.metrics import r2_score

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # PyTorch 1.9.0 이상에서 MPS 지원 확인
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def calculate_correlation(preds, labels):
    # 텐서를 CPU로 이동시키고 NumPy 배열로 변환
    preds_flat = preds.view(preds.size(0), -1).cpu().detach().numpy()
    labels_flat = labels.view(labels.size(0), -1).cpu().detach().numpy()
    
    # 배치별 상관관계 계산
    correlation_scores = []
    for i in range(preds_flat.shape[0]):  # 수정: .size(0) 대신 .shape[0] 사용
        # NaN 값이나 무한대 값을 0으로 대체
        pred = np.nan_to_num(preds_flat[i])
        label = np.nan_to_num(labels_flat[i])
        # 표준편차가 0이면 상관계수 계산을 건너뜀
        if np.std(pred) == 0 or np.std(label) == 0:
            continue
        corr = np.corrcoef(pred, label)[0, 1]
        correlation_scores.append(corr)
    # 평균 상관관계 반환
    return np.mean(correlation_scores)


def calculate_r2_score(tensor_true, tensor_pred):
    # 텐서 평탄화
    tensor_true_flat = tensor_true.view(tensor_true.size(0), -1).cpu().detach().numpy()
    tensor_pred_flat = tensor_pred.view(tensor_pred.size(0), -1).cpu().detach().numpy()
    
    # 배치별 R² 점수 계산
    r2_scores = []
    for true, pred in zip(tensor_true_flat, tensor_pred_flat):
        # NaN 값이나 무한대 값을 0으로 대체
        true = np.nan_to_num(true)
        pred = np.nan_to_num(pred)
        
        # 표준편차가 0이면 R² 점수 계산을 건너뜀
        if np.std(true) == 0 or np.std(pred) == 0:
            continue
        
        r2 = r2_score(true, pred)
        r2_scores.append(r2)
    if len(r2_scores) == 0:
        return np.nan  # 또는 적절한 기본값
    return np.nanmean(r2_scores)


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

def batch_miou(label_preds, label_trues, num_class, device):
    label_trues = label_trues.long()  # 정수 타입으로 변환
    label_preds = label_preds.long()  # 정수 타입으로 변환
    
    """Calculate mIoU for a batch of predictions and labels"""
    hist = torch.zeros((num_class, num_class), device=device)  # 디바이스 지정
    for lt, lp in zip(label_trues, label_preds):
        lt = lt.to(device)  # 디바이스 이동
        lp = lp.to(device)  # 디바이스 이동
        hist += fast_hist(lt.flatten(), lp.flatten(), num_class)
    return compute_miou(hist)




# if __name__ == "__main__":
#     # 예제 데이터
#     n_class = 3  # 클래스 수
#     batch_size = 10
#     height, width = 224, 224  # 예제 이미지 크기

#     # 임의의 실제 라벨과 예측 라벨 생성
#     label_trues = torch.randint(0, n_class, (batch_size, height, width))
#     label_preds = torch.randint(0, n_class, (batch_size, height, width))

#     # 배치 mIoU 계산
#     miou = batch_miou(label_preds,label_trues, n_class)
#     print(f"Batch mIoU: {miou.item()}")


#     # 상관관계 계산
#     correlation = calculate_correlation(label_preds, label_trues)
#     print(f"Correlation: {correlation}")


#     pred_carbon = torch.rand((batch_size,1, height, width))
#     label_carbon = torch.rand((batch_size,1, height, width))
#     # R² 점수 계산
#     r2_score = calculate_r2_score(pred_carbon, label_carbon)
#     print(f"R² Score: {r2_score}")