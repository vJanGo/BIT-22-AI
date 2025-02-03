import os
import json
from PIL import Image

def create_coco_json(image_dir, label_dir, output_file):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    categories = [{"id": 1, "name": "weed"}, {"id": 2, "name": "mq"}]
    coco_format["categories"] = categories

    annotation_id = 1
    for image_id, image_name in enumerate(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace('.png', '.json'))  

        # 读取图像信息
        with Image.open(image_path) as img:
            width, height = img.size

        # 添加图像信息到COCO格式
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        # 读取标签信息
        with open(label_path, 'r') as f:
            data = json.load(f)

        for shape in data['shapes']:
            points = shape['points']
            if shape['label'] == 'mq':
                category_id = 2
            else:
                category_id = 1

            if shape['shape_type'] == 'circle':
                x_center = points[0][0]
                y_center = points[0][1]
                radius = ((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)**0.5
                bbox_width = bbox_height = 2 * radius
                x_min = x_center - bbox_width / 2
                y_min = y_center - bbox_height / 2

                # 添加注释信息到COCO格式
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })
                annotation_id += 1

    # 保存为JSON文件
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

# 使用示例
create_coco_json('train/images', 'train/labels', 'train_coco.json')
