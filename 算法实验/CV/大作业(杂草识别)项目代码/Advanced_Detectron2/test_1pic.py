import os
import cv2
import pandas as pd
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
# 配置模型
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "output/model_final.pth"  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # 设置阈值
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.00001  # 设置NMS阈值
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
predictor = DefaultPredictor(cfg)

# 读取图片
image_path = "test/images/243.png"
image = cv2.imread(image_path)

# 预测
outputs = predictor(image)
# 可视化预测结果
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("weed_test"), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite("prediction.jpg", out.get_image()[:, :, ::-1])


print(outputs)  