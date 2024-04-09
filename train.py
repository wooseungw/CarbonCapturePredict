import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from models.dpt import DPTSegmentationWithCarbon
from dataset import CarbonDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # PyTorch 1.9.0 이상에서 MPS 지원 확인
        return torch.device("mps")
    else:
        return torch.device("cpu")


# 이전 wandb 실행의 ID 있는 경우만
#wandb_run_id = "YOUR_PREVIOUS_WANDB_RUN_ID"

# WandbLogger 설정, 이전 실행 재개
wandb_logger = WandbLogger(
    project="CCP",
    log_model="all",
#    resume="allow",  # 이전 실행이 있으면 재개, 없으면 새 실행 시작
#    id=wandb_run_id,  # 이전 실행 ID
)

# 체크포인트 콜백 설정
checkpoint_callback_min_loss = ModelCheckpoint(
    monitor='val_loss',  # 검증 손실을 모니터링
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,  # 최소 손실을 가진 상위 k개의 모델을 저장
    mode='min',  # 'min' 모드는 손실이 최소일 때 저장
)

checkpoint_callback_last = ModelCheckpoint(
    dirpath="checkpoints",  # 체크포인트 저장 경로
    filename='model-{epoch:02d}-{val_loss:.2f}-last',
    save_last=True,  # 마지막 에폭의 체크포인트를 저장
)
# device = select_device()
# print(f"Selected device: {device}")
def train():
    # Trainer 객체 생성
    trainer = L.Trainer(
    logger=wandb_logger,  # Wandb 로거 사용
    #resume_from_checkpoint=checkpoint_path, 
    callbacks=[checkpoint_callback_min_loss, checkpoint_callback_last],  # 콜백 리스트에 추가
    max_epochs=30,  # 최대 에폭 수
    )
    model =DPTSegmentationWithCarbon()
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    fp = 'Dataset/Training/image/SN10_Forest_IMAGE'
    train_dataloader = DataLoader(CarbonDataset(transform=transform,mode='Train',folder_path=fp), batch_size=4, shuffle=True,persistent_workers=True,num_workers=8)
    val_dataloader = DataLoader(CarbonDataset(transform=transform,mode='Valid',folder_path=fp), batch_size=4,persistent_workers=True ,num_workers=8)

    # 모델 학습
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
if __name__ == "__main__":
    train()