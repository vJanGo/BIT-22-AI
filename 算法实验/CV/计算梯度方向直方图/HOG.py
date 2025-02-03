import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(magnitude, angle, bins):
    
    bin_edges = np.linspace(0, 360, bins + 1)  # 定义bin边界
    histogram = np.zeros(bins) 

    # 将角度值映射到bin中
    bin_indices = np.digitize(angle, bin_edges[:-1]) - 1  # 获取bin索引
    bin_indices[bin_indices == bins] = 0  # 处理边界值:最后一个边界360置为0，首位相连

    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            histogram[bin_indices[i, j]] += magnitude[i, j] # 累加计算

    return histogram

def process_image(image_path, bins=9):
    # 载入图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    

    # 计算图像的x,y梯度
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    # 计算角度
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    # 分块(2x2)
    h, w = image.shape
    block_h, block_w = h // 2, w // 2

    histograms = []
    for i in range(2):
        for j in range(2):
            # 提取块
            block_magnitude = magnitude[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
            block_angle = angle[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]

            # 计算块的梯度方向直方图
            hist = calculate_histogram(block_magnitude, block_angle, bins)
            histograms.append(hist)

    # 可视化
    plt.figure(figsize=(12, 8))

    # 原图
    plt.subplot(3, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    # 显示直方图
    for idx, hist in enumerate(histograms):
        plt.subplot(3, 2, idx + 2)
        plt.title(f"Block {idx + 1}")
        plt.bar(range(bins), hist, width=1, color='blue', edgecolor='black')
        plt.xlabel("Direction Bin")
        plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()

# 使用函数
image_path = "test.jpg"  # 替换为你的图像路径
process_image(image_path, bins=12) # 超参数
