from ultralytics import YOLO
from natsort import natsorted
import os
import cv2
import csv

cnt = 0 #统计写入的数据个数

# 加载模型
model = YOLO('weeddetection/runs/detect/train4/weights/best.pt')

# 定义测试集文件夹路径
test_folder = 'weeddetection/test/images'

# 定义输出的 CSV 文件路径
output_csv = 'weeddetection/output/predictions4.csv'

# 创建 CSV 文件并写入表头
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'image_id', 'class_id', 'x_min', 'y_min', 'width', 'height'])  # 写入表头
    filelist = natsorted(os.listdir(test_folder))
    # 遍历测试集文件夹中的所有图片
    id = 1
    for idx, image_file in enumerate(filelist):
        image_path = os.path.join(test_folder, image_file)
        img = cv2.imread(image_path)

        # 进行图片预测
        results = model(img)

        # 遍历每张图片的结果
        for result in results:
                print(result.orig_shape)
                # 获取图片的ID（即文件名，不包括扩展名）
                image_id = os.path.splitext(image_file)[0]

                # 遍历每个检测框
                for box in result.boxes:
                    # 提取bounding box坐标并转换为numpy数组
                    xyxy = box.xyxy.cpu().numpy()[0]
                    xmin, ymin, xmax, ymax = xyxy[:4]
                    # print(xyxy)
                    # 获取分类ID
                    cls = int(box.cls.item())  # 类别
                    
                    height = abs(xmax - xmin)
                    weight = abs(ymax - ymin)
                    
                    # 写入 CSV 文件
                    writer.writerow([id, image_id, cls, xmin, ymin, weight, height])
                    id += 1
                    cnt += 1
    print(f"检测到的数据个数为 {cnt}")
    for i in range(5000 -  cnt - 1):
        writer.writerow([id, 99999, 9, 0, 0, 0, 0])
        id += 1
print(f"所有预测结果已保存到 {output_csv}")
#print(filelist)
