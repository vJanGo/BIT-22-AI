# 构建detect2数据集
import os
import json
from detectron2.structures import BoxMode
import cv2

def get_weed_dataset_dicts(img_dir, label_dir):
    dataset_dicts = []
    for filename in os.listdir(img_dir):
        record = {}
        img_path = os.path.join(img_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.png', '.json'))  #寻找json   

        # 读取标签文件
        with open(label_path) as f:
            data = json.load(f)
        record["file_name"] = img_path
        record["image_id"] = filename
        height, width = cv2.imread(img_path).shape[:2]
        record["height"] = height
        record["width"] = width

        objs = []
        for shape in data['shapes']:
            points = shape['points']
            if shape['label'] == 'mq':
                class_id = 1
            else:
                continue
            
            if shape['shape_type'] == 'circle':
                x_center = points[0][0]
                y_center = points[0][1]
                radius = ((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)**0.5
                width = height = 2 * radius
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
            
            
            
            
            
            obj = {
                "bbox": [x_min, y_min, x_max, y_max],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_id,
            }
            objs.append(obj)
        record["annotations"] = objs

        dataset_dicts.append(record)
    return dataset_dicts

# test
if __name__ == "__main__":
    img_dir = 'Meta_weeddetect/train/images'
    label_dir = 'Meta_weeddetect/train/labels'
    dataset_dicts = get_weed_dataset_dicts(img_dir, label_dir)
    print(dataset_dicts[1]['annotations'])
    print(dataset_dicts[1]['image_id'])