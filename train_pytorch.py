from models.dpt import DPTSegmentationWithCarbon
from dataset import CarbonDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn.functional as F
import wandb
import os
from models.util import select_device, batch_miou, calculate_r2_score, calculate_correlation
from  tqdm import tqdm 

def loss_fn(gt_pred, carbon_pred, gt, carbon, num_classes, device):
    # 학습 단계에서의 로스 계산 및 로깅
    gt_loss = F.cross_entropy(gt_pred, gt)
    if carbon_pred.dim() > carbon.dim():
        carbon_pred = carbon_pred.squeeze(1)  # 차원 축소
    carbon_loss = F.mse_loss(carbon_pred, carbon)
    
    _, gt_pred = torch.max(gt_pred, dim=1)
    miou = batch_miou(gt_pred, gt, num_classes, device)
    corr = calculate_correlation(carbon_pred, carbon)
    r2 = calculate_r2_score(carbon_pred, carbon)
    loss = gt_loss + (carbon_loss*0.005)
    return loss, gt_loss, carbon_loss, miou, corr, r2
    

#================================================================================================#
FOLDER_PATH={
    'Dataset/Training/image/AP10_Forest_IMAGE':7,
    'Dataset/Training/image/AP25_Forest_IMAGE':7,
    'Dataset/Training/image/AP10_City_IMAGE':9,
    'Dataset/Training/image/AP25_City_IMAGE':9,
    'Dataset/Training/image/SN10_Forest_IMAGE':4,
}
def train():
    fp = 'Dataset/Training/image/SN10_Forest_IMAGE'
    kwargs = {
        'num_classes': FOLDER_PATH[fp]
    }
    wandb.init(
        project="CCP",
        resume = True)
    wandb.config.update({
        'lr': 1e-4,
        'batch_size': 4,
        'num_epochs': 30,
        'num_classes': FOLDER_PATH[fp],
        'dir': fp
        })
    device = select_device()

    # 체크포인트를 저장할 경로 설정
    checkpoint_path = "checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)

    model =DPTSegmentationWithCarbon(**kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.lr)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train_dataloader = DataLoader(CarbonDataset(transform=transform,mode='Train',folder_path=fp), batch_size=wandb.config.batch_size, shuffle=True,num_workers=8)
    val_dataloader = DataLoader(CarbonDataset(transform=transform,mode='Valid',folder_path=fp), batch_size=wandb.config.batch_size, shuffle=True,num_workers=8)

    # 학습 및 검증 루프
    for epoch in tqdm(wandb.config.num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            x, carbon, gt = batch
            x, carbon, gt = x.to(device), carbon.to(device), gt.to(device)
            
            optimizer.zero_grad()  # 옵티마이저의 그래디언트를 0으로 초기화
            gt_pred, carbon_pred = model(x)  # 모델로부터 예측값을 얻음
            
            loss, gt_loss, carbon_loss, miou, corr,r2 = loss_fn(gt_pred, carbon_pred, gt, carbon, wandb.config.num_classes, device)
            wandb.log({"train_gt_loss": gt_loss,
                       "train_MSE,loss": carbon_loss,
                       "train_miou": miou,
                       "train_corr": corr,
                       "train_r2": r2
                    })
    
            loss.backward()  # 손실에 대한 모델의 그래디언트를 계산
            optimizer.step()  # 모델의 파라미터 업데이트
            
            train_loss += loss.item() * x.size(0)
            
            # WandB에 학습 손실 로깅
            wandb.log({"train_loss": loss.item()})
        
        # 에폭당 평균 학습 손실
        train_loss = train_loss / len(train_dataloader.dataset)
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in train_dataloader:
                x, carbon, gt = batch
                x, carbon, gt = x.to(device), carbon.to(device), gt.to(device)
                
                optimizer.zero_grad()  # 옵티마이저의 그래디언트를 0으로 초기화
                gt_pred, carbon_pred = model(x)  # 모델로부터 예측값을 얻음
                
                loss, gt_loss, carbon_loss, miou, corr,r2 = loss_fn(gt_pred, carbon_pred, gt, carbon, wandb.config.num_classes, device)
                wandb.log({
                    "Validation_gt_loss": gt_loss,
                    "Validation_MSE,loss": carbon_loss,
                    "Validation_miou": miou,
                    "Validation_corr": corr,
                    "Validation_r2": r2
                })
                val_loss += loss.item() * x.size(0)
            
                # WandB에 검증 손실 로깅
                wandb.log({"Validatation_Loss": loss.item()})
        
        # 에폭당 평균 검증 손실
        val_loss = val_loss / len(val_dataloader.dataset)
        # 검증 손실이 이전 최소값보다 낮으면 모델 저장
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f"model_epoch_{epoch}_val_loss_{val_loss:.4f}.pth"))
            print(f"Epoch {epoch}: Validation loss improved, saving model...")
        
        # 에폭별 학습 및 검증 손실 로깅
        wandb.log({"epoch": epoch, "average_train_loss": train_loss, "average_val_loss": val_loss})
        
    torch.save(model.state_dict(), os.path.join(checkpoint_path, f"model_last.pth"))
    print(f"Epoch {wandb.config.num_epochs}: Validation loss improved, saving model...")



if __name__ == "__main__":
    train()
    # 학습이 끝난 후 WandB 세션 종료
    wandb.finish()