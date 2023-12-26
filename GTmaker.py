import json
import numpy as np
from PIL import Image
from matplotlib.path import Path
from pyproj import Proj, transform
import os

def get_min_coordinates(features):
    min_x, min_y = float('inf'), float('inf')
    for feature in features:
        coords = feature['geometry']['coordinates'][0]
        for x, y in coords:
            min_x, min_y = min(min_x, x), min(min_y, y)
    return min_x, min_y

def transform_coordinates(coords, in_proj, out_proj):
    lon, lat = transform(in_proj, out_proj, coords[:, 0], coords[:, 1])
    return np.column_stack((lon, lat))

def create_segmentation_mask(features, mask_size, in_proj, out_proj):
    mask = np.zeros(mask_size, dtype=np.uint8)
    for feature in features:
        ann_cd = feature['properties']['ANN_CD']
        if ann_cd == 0:
            ann_cd = 100
        polygon_coords = np.array(feature['geometry']['coordinates'][0])
        
        # 좌표 변환
        transformed_coords = transform_coordinates(polygon_coords, in_proj, out_proj)
        
        # Path 객체를 사용하여 마스크에 그리기
        path = Path(transformed_coords)
        yy, xx = np.mgrid[:mask_size[0], :mask_size[1]]
        points = np.column_stack((xx.flatten(), yy.flatten()))
        mask[points[path.contains_points(points)]] = ann_cd

    return mask

def process_json_file(file_path, img_size, in_proj, out_proj):
    with open(file_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)

    min_x, min_y = get_min_coordinates(geojson_data['features'])
    mask = create_segmentation_mask(geojson_data['features'], img_size, in_proj, out_proj)
    return mask

# 예시 사용
folder_path = 'Dataset\Training\label\SN10_Forest_Json'
img_size = (512, 512)  # 이미지 크기
in_proj = Proj(init='epsg:5186')
out_proj = Proj(init='epsg:4326')  # EPSG 코드에 따라 적절한 값 사용

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        mask = process_json_file(file_path, img_size, in_proj, out_proj)
        
        # 이미지 크기에 맞게 resize (여기선 이미지 크기에 따라 resize 필요)
        resized_mask = Image.fromarray(mask).resize(img_size[::-1], resample=Image.Resampling.NEAREST)
        
        output_filename = filename.replace('.json', '.png')
        resized_mask.save(os.path.join(folder_path, output_filename))
