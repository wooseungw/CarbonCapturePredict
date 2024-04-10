import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from models.dpt import DPTSegmentationWithCarbon
from dataset import CarbonDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch



# 이전 wandb 실행의 ID 있는 경우만
#wandb_run_id = "YOUR_PREVIOUS_WANDB_RUN_ID"



# 체크포인트 콜백 설정
checkpoint_callback_min_loss = ModelCheckpoint(
    monitor='Validation_gt_loss',  # 검증 손실을 모니터링
    dirpath="checkpoints",  # 체크포인트 저장 경로
    filename='model-{epoch:02d}-{Validation_gt_loss:.2f}',
    save_top_k=1,  # 최소 손실을 가진 상위 k개의 모델을 저장
    mode='min',  # 'min' 모드는 손실이 최소일 때 저장
)

checkpoint_callback_last = ModelCheckpoint(
    dirpath="checkpoints",  # 체크포인트 저장 경로
    filename='model-{epoch:02d}-{val_loss:.2f}-last',
    save_last=True,  # 마지막 에폭의 체크포인트를 저장
)
FOLDER_PATH={
    'Dataset/Training/image/AP10_Forest_IMAGE':7,
    'Dataset/Training/image/AP25_Forest_IMAGE':7,
    'Dataset/Training/image/AP10_City_IMAGE':9,
    'Dataset/Training/image/AP25_City_IMAGE':9,
    'Dataset/Training/image/SN10_Forest_IMAGE':4,
}
# device = select_device()
# print(f"Selected device: {device}")
def train():
    fp = "Dataset/Training/image/AP10_Forest_IMAGE"
    num_classes = FOLDER_PATH[fp]
    # WandbLogger 설정, 이전 실행 재개
    wandb_logger = WandbLogger(
    project="CCP",
    log_model="all",  # 모든 모델 아티팩트 로깅
#    resume="allow",  # 이전 실행이 있으면 재개, 없으면 새 실행 시작
#    id=wandb_run_id,  # 이전 실행 ID
    
    )
    # Trainer 객체 생성
    trainer = L.Trainer(
        logger=wandb_logger,  # Wandb 로거 사용
        log_every_n_steps=50, # 50번의 Step마다 로깅
        #resume_from_checkpoint=checkpoint_path,
        callbacks=[checkpoint_callback_min_loss, checkpoint_callback_last],  # 콜백 리스트에 추가
        max_epochs=30,  # 최대 에폭 수
    )
    model =DPTSegmentationWithCarbon(num_classes= num_classes,path=None)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    
    
    train_dataloader = DataLoader(CarbonDataset(transform=transform,mode='Train',folder_path=fp), batch_size=18, shuffle=True,persistent_workers=True,num_workers=8)
    val_dataloader = DataLoader(CarbonDataset(transform=transform,mode='Valid',folder_path=fp), batch_size=32,persistent_workers=True ,num_workers=8)

    # 모델 학습
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
if __name__ == "__main__":
    train()