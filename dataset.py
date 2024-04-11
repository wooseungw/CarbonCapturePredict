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
        print(path, "Done.",end='/')
    print()
    return image_paths

class Mapping():
    def __init__(self, folder_path):
        if "AP10_City" in folder_path:
            self.label_name = "AP10_City"
            self.label_mapping={0:   0,                      
                                110: 1,
                                120: 2,
                                130: 3,
                                140: 4,
                                210: 5,
                                220: 6,
                                230: 7, 
                                190: 8,
                                255: 0}
        if "AP25_City" in folder_path:
            self.label_name = "AP25_City"
            self.label_mapping={0:   0,                      
                                110: 1,
                                120: 2,
                                130: 3,
                                140: 4,
                                210: 5,
                                220: 6,
                                230: 7, 
                                190: 8,
                                255: 0}
        if "AP10_Forest" in folder_path:
            self.label_name = "AP10_Forest"
            self.label_mapping={0:  0,                      
                                110: 1,
                                120: 2,
                                130: 3,
                                140: 4,
                                150: 5,
                                190: 6, 
                                255: 0}
        if "AP25_Forest" in folder_path:
            self.label_name = "AP25_Forest"
            self.label_mapping={0: 0,                      
                                110:1,
                                120:2,
                                130:3,
                                140: 4,
                                150: 5,
                                190: 6, 
                                255: 0}
        if "SN10_Forest" in folder_path:
            self.label_name = "SN10_Forest"
            self.label_mapping={0:  0,                      
                                140: 1,
                                150: 2,
                                190: 3, 
                                255: 0}
    def __call__(self, img):
        return self.gt_mapping(img)

    def gt_mapping(self, img):
        # PIL 이미지를 NumPy 배열로 변환
        image_np = np.array(img)
        
        # 출력 배열 초기화 (입력 이미지와 동일한 크기)
        mapped_image_np = np.zeros_like(image_np)
        
        # 라벨 매핑 적용
        for original_label, mapped_label in self.label_mapping.items():
            mapped_image_np[image_np == original_label] = mapped_label
        
        # 매핑된 NumPy 배열을 다시 PIL 이미지로 변환 (필요한 경우)
        mapped_image = Image.fromarray(mapped_image_np)
        
        return mapped_image
    
class CarbonDataset(Dataset):
    def __init__(self, folder_path, image_transform=None,sh_transform=None,label_transform=None, mode = "Train"):
        if mode == "Valid":
            folder_path = folder_path.replace("Training",'Validation')
            print("Validation Set")
        else:
            print("Training Set")
        #이미지와 임분고 리스트 반환
        self.image_paths = get_image_paths(folder_path)
        self.sh_paths = get_image_paths(folder_path.replace("IMAGE","SH"))
        #탄소량과 Gt리스트 반환
        folder_path = folder_path.replace("image",'label')
        self.carbon_paths = get_image_paths(folder_path.replace("IMAGE","Carbon"))
        self.gt_paths = get_image_paths(folder_path.replace("IMAGE","GT"))
        self.image_transform = image_transform
        self.sh_transform = sh_transform
        self.label_transform = label_transform
        self.Mapping = Mapping(folder_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        sh_path = self.sh_paths[idx]
        sh = Image.open(sh_path).convert('L')
        
        carbon_path = self.carbon_paths[idx]
        carbon = Image.open(carbon_path)
        
        gt_paths = self.gt_paths[idx]
        gt = Image.open(gt_paths).convert('L')
        gt = self.Mapping(gt)
        #print(gt)
        
        if self.image_transform:
            image = self.image_transform(image)
            
        if self.sh_transform:
            sh = self.sh_transform(sh)
            
        if self.label_transform:
            carbon = self.label_transform(carbon)
            gt = self.label_transform(gt)
            
        gt = torch.tensor(np.array(gt), dtype=torch.long)

        carbon = torch.tensor(np.array(carbon), dtype=torch.float64)
        
        # Concatenate image and sh along the channel dimension
        image_sh = torch.cat((image, sh), dim=0)

        return image_sh , carbon , gt
# 시각화 코드 예시
def imshow(tensor, title=None):
    image = tensor.numpy().transpose((1, 2, 0))
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated

if __name__ == "__main__":
    # Set the folder path for the dataset
    folder_path = 'Dataset/Training/image/AP10_Forest_IMAGE'
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    transform_label = transforms.Compose([transforms.Resize((256//4, 256//4))])
    # Create an instance of the CustomImageDataset class
    dataset = CarbonDataset(folder_path,transform,transform,transform_label, mode = "Train")

    # Create a data loader for the dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    sample_index = 0
    # Iterate over the dataset and print the images and labels
    for image_sh, carbon, gt in dataloader:
        for i in range(len(image_sh)):
            #carbon
            # print(carbon[i].shape, carbon[i].type())
            # print(carbon.min(), carbon.max())
            # print(carbon)
            # plt.imshow(carbon[i].squeeze(), cmap='gray')
            # plt.title("Sample Carbon")
            # plt.show()
            #gt
            print("GT:",gt[i].shape, gt[i].type())
            print(gt.min(), gt.max())
            # print(gt)
            # plt.imshow(gt[i].squeeze(), cmap='gray')  # squeeze()는 1채널 이미지의 경우 채널 차원을 제거
            # plt.title("Sample GT")
            # plt.show()

        print(image_sh.shape, image_sh.type())

        print(carbon.shape, carbon.type())
        print(gt.shape, gt.type())
        
        import matplotlib.pyplot as plt

        # Select one sample from the dataset
        sample_index += 1
        # 시각화

