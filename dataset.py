import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tifffile as tiff
import numpy as np

def get_image_paths(path):
    image_paths = []
    for filename in os.listdir(path):
        if filename.endswith('.tif') or filename.endswith('.png'):
            image_paths.append(os.path.join(path, filename))
    if len(image_paths) != 0:
        print(path, "Done.")
    return image_paths

class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None, mode = "Train"):
        if mode == "Valid":
            folder_path = folder_path.replace("Training",'Validation')
        #이미지와 임분고 리스트 반환
        self.image_paths = get_image_paths(folder_path)
        self.sh_paths = get_image_paths(folder_path.replace("IMAGE","SH"))
        #탄소량과 Gt리스트 반환
        folder_path = folder_path.replace("image",'label')
        self.carbon_paths = get_image_paths(folder_path.replace("IMAGE","Carbon"))
        self.gt_paths = get_image_paths(folder_path.replace("IMAGE","GT"))

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = tiff.imread(image_path).astype(np.float32)
        
        sh_path = self.sh_paths[idx]
        sh = tiff.imread(sh_path).astype(np.float32)
        carbon_path = self.carbon_paths[idx]
        carbon = tiff.imread(carbon_path).astype(np.float32)
        gt_paths = self.gt_paths[idx]
        gt = cv2.imread(gt_paths,cv2.IMREAD_GRAYSCALE).astype(np.float32)

        if self.transform:
            image = self.transform(image)
            sh = self.transform(sh)
            carbon = self.transform(carbon)
            gt = self.transform(gt)

        
        return image , sh, carbon, gt