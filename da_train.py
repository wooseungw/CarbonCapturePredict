import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.segformer_simple import Segwithcarbon, Segformerwithcarbon
from dataset import CarbonDataset
from models.util import select_device, mix_patch
from tqdm import tqdm
from models.metrics import CarbonLoss

import wandb
import os

def main():
    
    FOLDER_PATH={
    'Dataset/Training/image/AP10_Forest_IMAGE':7,
    'Dataset/Training/image/AP25_Forest_IMAGE':7,   
    'Dataset/Training/image/AP10_City_IMAGE':9,
    'Dataset/Training/image/AP25_City_IMAGE':9,
    'Dataset/Training/image/SN10_Forest_IMAGE':4,
    }
    
    fp = "Dataset/Training/image/AP25_Forest_IMAGE"
    target_fp = "Dataset/Training/image/SN10_Forest_IMAGE"
    label_size = 256 // 2
    args = {
    #C
    'dims':             (64, 128, 320, 512),
    'decoder_dim': 512,
    #R
    'reduction_ratio': (8, 4, 2, 1),
    #N
    'heads':           (1, 2, 5, 8),
    #E
    'ff_expansion':     (8, 8, 4, 4),
    #L
    'num_layers':       (2, 2, 2, 2),
    'channels': 4,#input channels
    'num_classes': FOLDER_PATH[fp],
    'stage_kernel_stride_pad': [(4, 2, 1), 
                                   (3, 2, 1), 
                                   (3, 2, 1), 
                                   (3, 2, 1)],
        'num_classes': FOLDER_PATH[fp],
    }

    epochs = 300
    lr = 1e-4
    device = select_device()
    batch_size = 1
    cls_lambda = 1
    reg_lambda = 0.0005
    source_dataset_name = fp.split("/")[-1]
    target_dataset_name = target_fp.split("/")[-1]
    model_name = "Segwithcarbon"
    checkpoint_path = f"checkpoints/{model_name}/Domain_Apdaptation"
    name = f"DA_{model_name}"+source_dataset_name.replace("_IMAGE", "")+f"_{label_size}"
    pretrain = None
    # Create the directory if it doesn't exist
    os.makedirs(checkpoint_path, exist_ok=True)
    wandb.login()
    wandb.init(
    # set the wandb project where this run will be logged
    project="CCP",
    name = name,
    # track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "fp": fp,
        "model_name": model_name,
        "cls_lambda": cls_lambda,
        "reg_lambda": reg_lambda,
        "checkpoint_path": checkpoint_path,
        "pretrain": pretrain,
        },
    )
    wandb.config.update(args)
    # 하이퍼파라미터 설정

    # 데이터셋을 위한 변환 정의
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    sh_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    label_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
    ])

    resizer = transforms.Compose([
        transforms.Resize((label_size, label_size))
    ])

    # 데이터셋 및 데이터 로더 생성
    train_dataset = CarbonDataset(fp, image_transform, sh_transform, label_transform,mode="Train")
    val_dataset = CarbonDataset(fp, image_transform,sh_transform, label_transform,mode="Valid")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=10,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=10,pin_memory=True)
    
    target_dataset = CarbonDataset(target_fp, image_transform, sh_transform, label_transform,mode="Train")
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True,num_workers=10,pin_memory=True)
    target_val_dataset = CarbonDataset(target_fp, image_transform,sh_transform, label_transform,mode="Valid")
    target_val_loader = DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False,num_workers=10,pin_memory=True)
     
    # 모델 생성
    if model_name == "Segwithcarbon":
        model = Segwithcarbon(**args)
    if model_name == "Segformerwithcarbon":
        model = Segformerwithcarbon(**args)
    if pretrain != None:
        model.load_state_dict(torch.load(pretrain), strict=False)
    model.to(device)
    #model = UNet_carbon(FOLDER_PATH[fp],dropout=True).to(device)
    # 손실 함수 및 옵티마이저 정의
    #gt_criterion = nn.CrossEntropyLoss(torch.tensor([0.] + [1.] * (FOLDER_PATH[fp]-1), dtype=torch.float)).to(device)
    loss = CarbonLoss(num_classes=FOLDER_PATH[fp],cls_lambda=cls_lambda,reg_lambda=reg_lambda).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # 학습
    glob_val_loss = 9e15
    for epoch in (range(epochs)):
        model.train()
        for (x, carbon, gt) ,(x_t,carbon_t,gt_t)in tqdm(zip(train_loader,target_loader), desc=f"Training Epoch {epoch+1}"):
            assert gt.min() >= 0 and gt.max() < FOLDER_PATH[fp], "라벨 값이 유효한 범위를 벗어났습니다."

            x = torch.cat((x, x_t), dim=0)
            random_index = torch.randint(1, x.size(2), (x.size(2)//2,))
            new = mix_patch(x, random_index, dataset_num=2, kernel_size=16)
            x = torch.cat((x,new),dim=0).to(device)

            
            carbon = torch.cat((carbon, carbon_t), dim=0)
            new = mix_patch(carbon,random_index,dataset_num=2,kernel_size=16)
            carbon = torch.cat((carbon,new),dim=0)
            carbon = resizer(carbon).to(device)

            
            gt = torch.cat((gt, gt_t), dim=0)
            new = mix_patch(gt,random_index,dataset_num=2,kernel_size=16)
            gt = torch.cat((gt,new),dim=0)
            gt = resizer(gt).to(device)

            new = None
            
            optimizer.zero_grad()
            gt_pred, carbon_pred  = model(x)
            #print(gt_pred.shape, gt_pred.type, gt.squeeze(1).shape, carbon_pred.shape, carbon.shape)
            total_loss, cls_loss, reg_loss, acc_c, acc_r, miou = loss(gt_pred, gt.squeeze(1), carbon_pred, carbon)
            #total_loss = gt_criterion(gt_pred, gt.squeeze(1))
            
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Train Loss: {total_loss.item():.4f}, Train cls_loss: {cls_loss.item():.4f}, Train reg_loss: {reg_loss.item():.4f}, Train acc_c: {acc_c:.4f}, Train acc_r: {acc_r:.4f} , Train miou: {miou:.4f}")
        wandb.log({"Train Loss":total_loss.item(), "Train cls_loss":cls_loss.item(), "Train reg_loss":reg_loss.item(), "Train acc_c":acc_c, "Train acc_r":acc_r, "Train miou":miou})
        val_total_loss = 0
        model.eval()
        for  (x, carbon, gt) ,(x_t,carbon_t,gt_t) in tqdm(zip(val_loader,target_val_loader), desc=f"Validation Epoch {epoch+1}"):
            #x = torch.cat((image, sh), dim=0)
            x = torch.cat((x, x_t), dim=0).to(device)
            carbon = resizer(torch.cat((carbon, carbon_t), dim=0)).to(device)
            gt = resizer(torch.cat((gt, gt_t), dim=0)).to(device)
            
            gt_pred, carbon_pred  = model(x)
            total_loss, cls_loss, reg_loss, acc_c, acc_r, miou = loss(gt_pred, gt.squeeze(1), carbon_pred, carbon)
            #total_loss = gt_criterion(gt_pred, gt.squeeze(1))
            
            val_total_loss += total_loss.item()
        val_total_loss /= (len(val_loader)+len(target_val_loader))  # 평균 검증 손실 계산
        if val_total_loss < glob_val_loss:
            glob_val_loss = val_total_loss
            torch.save(model.state_dict(), f"{checkpoint_path}/{name}_best.pth")

        print(f"Validation Loss: {val_total_loss:.4f}, Validation cls_loss: {cls_loss.item():.4f}, Validation reg_loss: {reg_loss.item():.4f}, Validation acc_c: {acc_c:.4f}, Validation acc_r: {acc_r:.4f}, Validation miou: {miou:.4f}")
        wandb.log({"Validation Loss":val_total_loss, "Validation cls_loss":cls_loss.item(), "Validation reg_loss":reg_loss.item(), "Validation acc_c":acc_c, "Validation acc_r":acc_r , "Validation miou":miou})
        wandb.log({"Epoch":epoch+1})
    torch.save(model.state_dict(), f"{checkpoint_path}/{name}_last_{epoch+1}.pth")
    wandb.finish()
if __name__ =="__main__":
    
    
    main()