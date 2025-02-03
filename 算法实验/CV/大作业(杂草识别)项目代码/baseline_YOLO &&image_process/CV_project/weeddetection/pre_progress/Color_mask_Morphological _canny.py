import cv2
import os
import numpy as np

def process_image(input_path, output_path):
    """
    处理单张图像：颜色分离 -> 形态学操作 -> Canny 边缘检测
    """
    # 读取图像
    image = cv2.imread(input_path)

    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义绿色范围的HSV阈值
    lower_green = np.array([35, 40, 40])  # 绿色范围下限
    upper_green = np.array([85, 255, 255])  # 绿色范围上限

    # 颜色分离：提取绿色区域
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 形态学操作：膨胀和腐蚀
    kernel = np.ones((5, 5), np.uint8)  # 定义卷积核大小
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)  # 膨胀
    mask_eroded = cv2.erode(mask_dilated, kernel, iterations=1)  # 腐蚀

    # 用形态学处理后的掩膜提取绿色区域
    green_segmented = cv2.bitwise_and(image, image, mask=mask_eroded)

    # 转换为灰度图以便 Canny 检测
    gray = cv2.cvtColor(green_segmented, cv2.COLOR_BGR2GRAY)

    # Canny 边缘检测
    edges = cv2.Canny(gray, 100, 200)

    # 保存结果
    output_segment_path = os.path.join(output_path, "" + os.path.basename(input_path))
    output_edge_path = os.path.join(output_path, "edges_" + os.path.basename(input_path))
    output_morph_path = os.path.join(output_path, "morph_" + os.path.basename(input_path))

    # 保存绿色分离结果和边缘检测结果
    cv2.imwrite(output_segment_path, green_segmented)
    # cv2.imwrite(output_edge_path, edges)
    # cv2.imwrite(output_morph_path, mask_eroded)  # 保存形态学处理后的掩膜图


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
