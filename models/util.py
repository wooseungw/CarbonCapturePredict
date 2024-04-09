import torch

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