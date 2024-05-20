import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models.segformer_simple import Segwithcarbon, Segformerwithcarbon
from dataset import CarbonDataset
from models.util import select_device, mix_patch
from models.metrics import CarbonLoss

def initialize_wandb(name, config, args):
    wandb.login()
    wandb.init(project="CCP", name=name, config=config)
    wandb.config.update(args)

def get_data_loaders(fp, target_fp, batch_size, label_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    label_transform = transforms.Compose([
        transforms.Resize((256, 256)),
    ])
    
    resizer = transforms.Compose([
        transforms.Resize((label_size, label_size))
    ])

    train_dataset = CarbonDataset(fp, transform, transform, label_transform, mode="Train")
    val_dataset = CarbonDataset(fp, transform, transform, label_transform, mode="Valid")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    target_dataset = CarbonDataset(target_fp, transform, transform, label_transform, mode="Train")
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    target_val_dataset = CarbonDataset(target_fp, transform, transform, label_transform, mode="Valid")
    target_val_loader = DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader, target_loader, target_val_loader, resizer

def train_one_epoch(model, train_loader, target_loader, optimizer, loss_fn, device, resizer):
    model.train()
    train_total_loss = 0.0
    train_total_cls_loss = 0.0
    train_total_reg_loss = 0.0
    train_total_acc_c = 0.0
    train_total_acc_r = 0.0
    train_total_miou = 0.0
    train_batches = 0

    for (x, carbon, gt), (x_t, carbon_t, gt_t) in tqdm(zip(train_loader, target_loader), desc="Training"):
        assert gt.min() >= 0 and gt.max() < 5, "라벨 값이 유효한 범위를 벗어났습니다."

        x = torch.cat((x, x_t), dim=0).to(device)
        carbon = resizer(torch.cat((carbon, carbon_t), dim=0)).to(device)
        gt = resizer(torch.cat((gt, gt_t), dim=0)).to(device)

        optimizer.zero_grad()
        gt_pred, carbon_pred = model(x)
        total_loss, cls_loss, reg_loss, acc_c, acc_r, miou = loss_fn(gt_pred, gt.squeeze(1), carbon_pred, carbon)
        total_loss.backward()
        optimizer.step()

        train_total_loss += total_loss.item()
        train_total_cls_loss += cls_loss.item()
        train_total_reg_loss += reg_loss.item()
        train_total_acc_c += acc_c
        train_total_acc_r += acc_r
        train_total_miou += miou
        train_batches += 1

    avg_train_loss = train_total_loss / train_batches
    avg_train_cls_loss = train_total_cls_loss / train_batches
    avg_train_reg_loss = train_total_reg_loss / train_batches
    avg_train_acc_c = train_total_acc_c / train_batches
    avg_train_acc_r = train_total_acc_r / train_batches
    avg_train_miou = train_total_miou / train_batches

    return avg_train_loss, avg_train_cls_loss, avg_train_reg_loss, avg_train_acc_c, avg_train_acc_r, avg_train_miou

def validate_one_epoch(model, val_loader, target_val_loader, loss_fn, device, resizer):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_acc_c = 0.0
    total_acc_r = 0.0
    total_miou = 0.0
    total_batches = 0

    with torch.no_grad():
        for (x, carbon, gt), (x_t, carbon_t, gt_t) in tqdm(zip(val_loader, target_val_loader), desc="Validation"):
            x = torch.cat((x, x_t), dim=0).to(device)
            carbon = resizer(torch.cat((carbon, carbon_t), dim=0)).to(device)
            gt = resizer(torch.cat((gt, gt_t), dim=0)).to(device)
            
            gt_pred, carbon_pred = model(x)
            loss, cls_loss, reg_loss, acc_c, acc_r, miou = loss_fn(gt_pred, gt.squeeze(1), carbon_pred, carbon)
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            total_acc_c += acc_c
            total_acc_r += acc_r
            total_miou += miou
            total_batches += 1

    avg_loss = total_loss / total_batches
    avg_cls_loss = total_cls_loss / total_batches
    avg_reg_loss = total_reg_loss / total_batches
    avg_acc_c = total_acc_c / total_batches
    avg_acc_r = total_acc_r / total_batches
    avg_miou = total_miou / total_batches

    return avg_loss, avg_cls_loss, avg_reg_loss, avg_acc_c, avg_acc_r, avg_miou

def main():
    FOLDER_PATH = {
        'Dataset/Training/image/AP10_Forest_IMAGE': 4,
        'Dataset/Training/image/AP25_Forest_IMAGE': 4,   
        'Dataset/Training/image/AP10_City_IMAGE': 4,
        'Dataset/Training/image/AP25_City_IMAGE': 4,
        'Dataset/Training/image/SN10_Forest_IMAGE': 4,
    }

    fp = "Dataset/Training/image/AP25_Forest_IMAGE"
    target_fp = "Dataset/Training/image/AP25_City_IMAGE"
    label_size = 256 // 2
    args = {
        'dims': (64, 128, 320, 512),
        'decoder_dim': 512,
        'reduction_ratio': (8, 4, 2, 1),
        'heads': (1, 2, 5, 8),
        'ff_expansion': (8, 8, 4, 4),
        'num_layers': (3, 3, 8, 3),
        'channels': 4,  # input channels
        'num_classes': FOLDER_PATH[fp],
        'stage_kernel_stride_pad': [
            (4, 2, 1), 
            (3, 2, 1), 
            (3, 2, 1), 
            (3, 2, 1)
        ],
    }

    epochs = 200
    lr = 1e-4
    device = select_device()
    batch_size = 1
    cls_lambda = 1
    reg_lambda = 0.005
    source_dataset_name = fp.split("/")[-1]
    target_dataset_name = target_fp.split("/")[-1]
    model_name = "Segformerwithcarbon"
    checkpoint_path = f"checkpoints/{model_name}/Domain_Apdaptation"
    name = f"DA_B3_{model_name}_{source_dataset_name.replace('_IMAGE', '')}_{label_size}"
    pretrain = None

    os.makedirs(checkpoint_path, exist_ok=True)

    initialize_wandb(name, {
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "fp": fp,
        "model_name": model_name,
        "cls_lambda": cls_lambda,
        "reg_lambda": reg_lambda,
        "checkpoint_path": checkpoint_path,
        "pretrain": pretrain,
    }, args)

    train_loader, val_loader, target_loader, target_val_loader, resizer = get_data_loaders(fp, target_fp, batch_size, label_size)

    if model_name == "Segwithcarbon":
        model = Segwithcarbon(**args)
    elif model_name == "Segformerwithcarbon":
        model = Segformerwithcarbon(**args)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    if pretrain is not None:
        model.load_state_dict(torch.load(pretrain), strict=False)

    model.to(device)
    loss_fn = CarbonLoss(num_classes=FOLDER_PATH[fp], cls_lambda=cls_lambda, reg_lambda=reg_lambda).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    glob_val_loss = float('inf')

    for epoch in range(epochs):
        avg_train_loss, avg_train_cls_loss, avg_train_reg_loss, avg_train_acc_c, avg_train_acc_r, avg_train_miou = train_one_epoch(
            model, train_loader, target_loader, optimizer, loss_fn, device, resizer)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train cls_loss: {avg_train_cls_loss:.4f}, Train reg_loss: {avg_train_reg_loss:.4f}")
        print(f"Train carbon Acc: {avg_train_acc_c:.4f}, Train roughness Acc: {avg_train_acc_r:.4f}, Train mIoU: {avg_train_miou:.4f}")
        avg_val_loss, avg_val_cls_loss, avg_val_reg_loss, avg_val_acc_c, avg_val_acc_r, avg_val_miou = validate_one_epoch(
            model, val_loader, target_val_loader, loss_fn, device, resizer)

        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val cls_loss: {avg_val_cls_loss:.4f}, Val reg_loss: {avg_val_reg_loss:.4f}")
        print(f"Val carbon Acc: {avg_val_acc_c:.4f}, Val roughness Acc: {avg_val_acc_r:.4f}, Val mIoU: {avg_val_miou:.4f}")

        wandb.log({
            "train/total_loss": avg_train_loss,
            "train/cls_loss": avg_train_cls_loss,
            "train/reg_loss": avg_train_reg_loss,
            "train/acc_c": avg_train_acc_c,
            "train/acc_r": avg_train_acc_r,
            "train/miou": avg_train_miou,
            "val/total_loss": avg_val_loss,
            "val/cls_loss": avg_val_cls_loss,
            "val/reg_loss": avg_val_reg_loss,
            "val/acc_c": avg_val_acc_c,
            "val/acc_r": avg_val_acc_r,
            "val/miou": avg_val_miou,
        })

        if avg_val_loss < glob_val_loss:
            glob_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{checkpoint_path}/best_model.pt")
            print(f"Best model saved with val loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), f"{checkpoint_path}/{name}_last_{epoch+1}.pth")
    wandb.finish()