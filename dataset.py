import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_image_paths(path):
    image_paths = []
    for filename in os.listdir(path):
        if filename.endswith('.tif') or filename.endswith('.png'):
            image_paths.append(os.path.join(path, filename))
    if len(image_paths) != 0:
        print(path, "Done.")
    return image_paths

class CarbonDataset(Dataset):
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
        image = Image.open(image_path).convert('RGB')
        
        sh_path = self.sh_paths[idx]
        sh = Image.open(sh_path).convert('L')
        carbon_path = self.carbon_paths[idx]
        carbon = Image.open(carbon_path).convert('L')
        gt_paths = self.gt_paths[idx]
        gt = Image.open(gt_paths).convert('L')
        if self.transform:
            image = self.transform(image)
            sh = self.transform(sh)
            carbon = self.transform(carbon)
            gt = self.transform(gt)
            

        # Concatenate image and sh along the channel dimension
        image_sh = torch.cat((image, sh), dim=0)

        return image_sh.long() , carbon.squeeze().long() , gt.squeeze().long()
# 시각화 코드 예시
def imshow(tensor, title=None):
    image = tensor.numpy().transpose((1, 2, 0))
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated

if __name__ == "__main__":
    # Set the folder path for the dataset
    folder_path = 'Dataset/Training/image/SN10_Forest_IMAGE'
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    # Create an instance of the CustomImageDataset class
    dataset = CarbonDataset(folder_path,transform=transform, mode = "Train")

    # Create a data loader for the dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    sample_index = 0
    # Iterate over the dataset and print the images and labels
    for image_sh, carbon, gt in dataloader:
        print(image_sh.shape)

        print(carbon.shape)
        print(gt.shape)
        break
        import matplotlib.pyplot as plt

        # Select one sample from the dataset
        sample_index += 1
        # 시각화
        imshow(sample_image, "Sample Image")
        imshow(sample_sh, "Sample SH")
        plt.imshow(sample_carbon.squeeze(), cmap='gray')
        plt.title("sample_carbon")
        plt.show()

        # 그레이스케일 이미지는 직접 시각화 가능
        plt.imshow(sample_gt.squeeze(), cmap='gray')  # squeeze()는 1채널 이미지의 경우 채널 차원을 제거
        plt.title("Sample GT")
        plt.show()
