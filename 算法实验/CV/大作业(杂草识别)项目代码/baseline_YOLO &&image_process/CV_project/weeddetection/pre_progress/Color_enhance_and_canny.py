# 设置输入和输出文件夹路径
# input_folder = "weeddetection/train/images"  
# output_folder = "weeddetection/progressed_img/train"  
import cv2
import os
import numpy as np

def process_image(input_path, output_path):
    """
    处理单张图像：增强绿色区域并提取边缘
    """
    # 读取图像
    image = cv2.imread(input_path)

    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])  # 绿色范围下限
    upper_green = np.array([85, 255, 255])  # 绿色范围上限
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(image, image, mask=mask)

    # 边缘检测
    edges = cv2.Canny(result, 100, 200)

    # 保存结果图像
    output_edge_path = os.path.join(output_path, "edges_" + os.path.basename(input_path))
    output_result_path = os.path.join(output_path, "" + os.path.basename(input_path))

    cv2.imwrite(output_result_path, result)
    cv2.imwrite(output_edge_path, edges)


def process_folder(input_folder, output_folder):
    """
    批量处理文件夹中的图像
    """
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # 检查文件是否为图像文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            print(f"Processing: {filename}")
            process_image(input_path, output_folder)
        else:
            print(f"Skipping non-image file: {filename}")

# 设置输入和输出文件夹路径
input_folder = "weeddetection/test/images"  
output_folder = "weeddetection/progressed_img/test"  

process_folder(input_folder, output_folder)
