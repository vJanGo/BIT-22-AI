# 输出结果

import os
import cv2
import pandas as pd
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from dataset_init import get_weed_dataset_dicts
import csv
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

for d in ['test']:
        # 加载测试集       
    DatasetCatalog.register("weed_" + d, lambda d=d: get_weed_dataset_dicts('test/images', 'test/labels'))
    MetadataCatalog.get("weed_" + d).set(thing_classes=["weed", "mq"])
# 配置模型
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "output/model_final.pth"  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 设置阈值
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.00001  # 设置NMS阈值
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

 
# 获取元数据
metadata = MetadataCatalog.get("weed_test")
predictor = DefaultPredictor(cfg)
# 创建评估器
evaluator = COCOEvaluator("weed_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "weed_test")

# 进行评估
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# 预测并保存结果
results = []
image_dir = "test/images"  # 图片文件夹路径
ID = 1

for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    outputs = predictor(image)
    # 获取预测结果
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    scores = instances.scores if instances.has("scores") else None
    classes = instances.pred_classes if instances.has("pred_classes") else None
    
    # 将结果添加到列表中
    for i in range(len(boxes)):
        box = boxes[i].tensor.numpy()[0]
        class_id = int(classes[i])
        class_name = metadata.thing_classes[class_id]  # 获取类别名称
        
        x_min, y_min, width, height = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
        cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), (0, 0, 0), -1)
        
        result = {
            "ID": ID,
            "image_id": image_name.replace('.png',''),
            "class_id": class_id,
            "x_min": x_min,
            "y_min": y_min,
            "width": width,
            "height": height
        }
        results.append(result)
        ID += 1


# 保存结果到CSV文件
df = pd.DataFrame(results)
df.to_csv("output/results/output_predictions_4.csv",  index=False)

# 补全到5000行
total_rows = 5000
current_length = len(df)
rows_to_add = total_rows - current_length

# 打开CSV文件以追加模式写入
with open("output/results/output_predictions_4.csv", 'a', newline='') as file:
    writer = csv.writer(file)
    
    # 从现有行数开始补全到5000行
    for i in range(current_length + 1, total_rows):
        writer.writerow([i, 99999, 9, 0, 0, 0, 0])
