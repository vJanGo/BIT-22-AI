import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """使用 OpenCV 加载图像并转换为 NumPy 数组"""
    return cv2.imread(image_path)

def match_template(big_image, small_image):
    """实现模板匹配，返回响应图"""
    big_h, big_w, _ = big_image.shape
    small_h, small_w, _ = small_image.shape
    
    # 初始化响应图
    response_map = np.zeros((big_h - small_h + 1, big_w - small_w + 1))

    # 计算小图的均值和标准差
    small_mean = np.mean(small_image)
    small_std = np.std(small_image)

    total_iterations = (big_h - small_h + 1) * (big_w - small_w + 1)

    for y in range(big_h - small_h + 1):
        for x in range(big_w - small_w + 1):
            # 获取当前窗口
            window = big_image[y:y + small_h, x:x + small_w]
            
            # 计算窗口的均值和标准差
            window_mean = np.mean(window)
            window_std = np.std(window)

            # 计算归一化相关性
            numerator = np.sum((window - window_mean) * (small_image - small_mean))
            denominator = window_std * small_std

            if denominator != 0:
                response_map[y, x] = numerator / denominator

            # 打印进度：每处理 1% 时更新进度条
            current_iteration = y * (big_w - small_w + 1) + x + 1
            if current_iteration % (total_iterations // 100) == 0:  # 每处理 1% 时打印
                progress = current_iteration / total_iterations * 100
                bar_length = 50  # 条形图的长度
                block = int(round(bar_length * progress / 100))
                progress_bar = f"\r进度: [{'#' * block}{'-' * (bar_length - block)}] {progress:.2f}%"
                sys.stdout.write(progress_bar)
                sys.stdout.flush()

    print()  # 换行以避免与后续输出重叠
    return response_map

    

def min_max_loc(response_map):
    """查找响应图中的最小值和最大值及其位置"""
    min_val = np.min(response_map)
    max_val = np.max(response_map)
    min_loc = np.unravel_index(np.argmin(response_map), response_map.shape)
    max_loc = np.unravel_index(np.argmax(response_map), response_map.shape)
    
    return min_val, max_val, min_loc, max_loc

# 载入彩色图像
big_image = load_image('big_image.jpg')
small_image = load_image('small_image.jpg')

# 计算响应图
response_map = match_template(big_image, small_image)

# 找到响应图中的最大值及其位置
min_val, max_val, min_loc, max_loc = min_max_loc(response_map)

# 在大图上标记匹配区域
top_left = max_loc
h, w, _ = small_image.shape
bottom_right = (top_left[0] + h, top_left[1] + w)

# 绘制矩形标记匹配区域
big_image_with_match = big_image.copy()
cv2.rectangle(big_image_with_match, top_left[::-1], bottom_right[::-1], (0, 0, 255), 2)

# 显示结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Big Image with Match')
plt.imshow(cv2.cvtColor(big_image_with_match, cv2.COLOR_BGR2RGB))  # 转换为 RGB
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Small Image (Template)')
plt.imshow(cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB))  # 转换为 RGB
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Response Map')
plt.imshow(response_map, cmap='hot')
plt.colorbar()
plt.axis('off')

plt.show()
