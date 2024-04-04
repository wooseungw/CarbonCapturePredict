import os

import PIL.Image
import PIL.ImageShow
from dataset import get_image_paths
import PIL
import numpy as np
import matplotlib.pyplot as plt
"""
    주소를 바탕으로 IMAGE와 SH 이미지를 가져온다.
    가져온 이미지와 임분고 정보를 numpy로 변환한다.
    두 넘파이를 합친후 이미지(png)로 저장한다.
    이 이미지는 4채널로 이루어져있다. (RGB+SH)
    따라서 원래 RGB이미지만 보고싶다면 넘파이로 변환하거나 텐서로 변환 후 앞의 3차원만 사용하면된다.
"""
dirs = 'Dataset/Training/image/SN10_Forest_IMAGE'

def process_image(dirs):
    paths = get_image_paths(dirs)
    sh_paths = get_image_paths(dirs.replace("IMAGE","SH"))
    for path, sh_path in zip(paths, sh_paths):
        im = PIL.Image.open(path).convert('RGB')
        sh = PIL.Image.open(sh_path).convert('L')
        im = np.array(im)
        sh = np.array(sh)
        new = np.concatenate((im, sh[..., np.newaxis]), axis=2)
        save_path = path.replace("image", "images").replace(".tif", ".png")
        new_dir = dirs.replace("image", "images")
        if new_dir not in os.listdir():
            os.mkdir(new_dir)
        new_image = PIL.Image.fromarray(new)
        new_image.save(save_path)
    print(f"Image saved at {save_path}")

#process_image(dirs)
print(os.listdir())