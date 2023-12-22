import os
import json
import cv2
import numpy as np
from pyproj import CRS, Transformer

def convert_crs(x, y):
    src_proj = CRS('epsg:5186')  # 원본 CRS
    dst_proj = CRS('epsg:4326')  # 타겟 CRS (WGS84)
    # Transformer 객체 생성
    transformer = Transformer.from_crs(src_proj, dst_proj)
    # 좌표 변환
    lon, lat = transformer.transform(x, y)
    return [lon, lat]

def convert_coordinates_to_pixel(coordinates, pixel_to_meter_ratio, image_size=(512, 512)):
    x, y = coordinates
    pixel_x = x * pixel_to_meter_ratio
    pixel_y = image_size[1] - y * pixel_to_meter_ratio
    return int(pixel_x), int(pixel_y)

def create_mask(filename, output_directory, pixel_to_meter_ratio, image_size=(512, 512)):
    with open(filename, 'r',encoding='utf-8') as f:
        data = json.load(f)

    polygons = []
    for feature in data['features']:
        if feature['properties']['ANN_CD'] == 0:
            feature['properties']['ANN_CD'] = 100
        # CRS 변환 후 픽셀 좌표로 변환
        polygon = [convert_crs(*point[::-1]) for point in feature['geometry']['coordinates'][0]]
        print("After CRS conversion:", polygon)  # 디버깅 코드 추가
        polygon = [convert_coordinates_to_pixel(point, pixel_to_meter_ratio, image_size) for point in polygon]
        print("After coordinate conversion:", polygon)  # 디버깅 코드 추가
        polygons.append((polygon, feature['properties']['ANN_CD']))

    mask = np.zeros(image_size, dtype=np.uint8)
    for polygon, ann_cd in polygons:
        print("Filling polygon with ANN_CD:", ann_cd)  # 디버깅 코드 추가
        cv2.fillPoly(mask, np.array([polygon], dtype=np.int32), color=(ann_cd))

        # 수정된 코드
    new_filename = os.path.splitext(os.path.basename(filename))[0]
    new_filename = new_filename.replace('.json', '') + '.png'
    cv2.imwrite(os.path.join(output_directory, new_filename), mask)
    
# 각 pixel_to_meter_ratio에 대해 마스크 생성
for directory_name, pixel_to_meter_ratio in zip(['Dataset\Training\label\SN10_Forest_Json', 'Dataset\Training\label\AP10_Forest_Json', 'Dataset\Training\label\AP25_Forest_Json'], [10, 0.1, 0.25]):
    os.makedirs(directory_name, exist_ok=True)
    for filename in os.listdir(directory_name):
        if filename.endswith('.json'):
            create_mask(os.path.join(directory_name, filename), directory_name, pixel_to_meter_ratio)
