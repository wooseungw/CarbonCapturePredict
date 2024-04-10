import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.segformer_simple import Segformer, Segformerwithcarbon
from dataset import CarbonDataset
from models.util import select_device
from tqdm import tqdm
from models.metrics import CarbonLoss
from models.unet import UNet_carbon
def main():
    # 하이퍼파라미터 설정
    FOLDER_PATH={
        'Dataset/Training/image/AP10_Forest_IMAGE':7,
        'Dataset/Training/image/AP25_Forest_IMAGE':7,   
        'Dataset/Training/image/AP10_City_IMAGE':9,
        'Dataset/Training/image/AP25_City_IMAGE':9,
        'Dataset/Training/image/SN10_Forest_IMAGE':4,
    }
    checkpoint_path = "checkpoint"
    fp = "Dataset/Training/image/AP10_Forest_IMAGE"
    epochs = 100
    lr = 1e-3
    device = select_device()
    batch_size = 32
    
    args = {
    'dims': (32, 64, 160, 256),
    'heads': (1, 2, 5, 8),
    'ff_expansion': (8, 8, 4, 4),
    'reduction_ratio': (8, 4, 2, 1),
    'num_layers': 2,
    'channels': 4,
    'decoder_dim': 256,
    'num_classes': FOLDER_PATH[fp]
    }
    
    # 데이터셋을 위한 변환 정의
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB 각 채널에 대해 평균 0.5, 표준편차 0.5 적용
    ])
    sh_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 그레이스케일 채널에 대해 평균 0.5, 표준편차 0.5 적용
    ])
    label_transform = transforms.Compose([
        transforms.Resize((256//4, 256//4)),
        transforms.ToTensor()
    ])
    # 데이터셋 및 데이터 로더 생성
    train_dataset = CarbonDataset(fp, image_transform, sh_transform, label_transform,mode="Train")
    val_dataset = CarbonDataset(fp, image_transform,sh_transform, label_transform,mode="Valid")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=8)
    # 모델 생성
    model = Segformerwithcarbon(**args).to(device)    
    #model = UNet_carbon(FOLDER_PATH[fp],dropout=True).to(device)
    # 손실 함수 및 옵티마이저 정의
    #gt_criterion = nn.CrossEntropyLoss(torch.tensor([0.] + [1.] * (FOLDER_PATH[fp]-1), dtype=torch.float)).to(device)
    loss = CarbonLoss(num_classes=FOLDER_PATH[fp]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # 학습
    glob_val_loss = 10000
    for epoch in tqdm(range(epochs)):
        model.train()
        for x, carbon, gt in tqdm(train_loader, desc="Training"):
            
            x, carbon, gt = x.to(device), carbon.to(device), gt.to(device)
            optimizer.zero_grad()
            gt_pred, carbon_pred  = model(x)
            total_loss, cls_loss, reg_loss, acc_c, acc_r = loss(gt_pred, gt.squeeze(1).long(), carbon_pred, carbon)
            
            total_loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss.item()}, cls_loss: {cls_loss.item()}, reg_loss: {reg_loss.item()}, acc_c: {acc_c}, acc_r: {acc_r}")
        val_total_loss = 0
        model.eval()
        for  x, carbon, gt in tqdm(val_loader, desc="Validation"):
            #x = torch.cat((image, sh), dim=0)
            x, carbon, gt = x.to(device), carbon.to(device), gt.to(device)
            gt_pred, carbon_pred  = model(x)
            total_loss, cls_loss, reg_loss, acc_c, acc_r = loss(gt_pred, gt.squeeze(1).long(), carbon_pred, carbon)
            val_total_loss += total_loss.item()
        val_total_loss /= len(val_loader)  # 평균 검증 손실 계산
        if val_total_loss < glob_val_loss:
            glob_val_loss = val_total_loss
            torch.save(model.state_dict(), f"{checkpoint_path}/best_model_{epoch+1}.pth")
        print(f"Validation Loss: {glob_val_loss}, Current Loss: {val_total_loss} , cls_loss: {cls_loss.item()}, reg_loss: {reg_loss.item()}, acc_c: {acc_c}, acc_r: {acc_r}")

if __name__ =="__main__":
    
    main()