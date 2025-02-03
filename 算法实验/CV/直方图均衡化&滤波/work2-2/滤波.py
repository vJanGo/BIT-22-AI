# 运行时间较长

import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文

# 加载彩色图像并归一化到 [0, 1] 范围
image = cv2.imread('OIP.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将 BGR 转为 RGB
image = image / 255.0  # 归一化

# 添加随机噪声
def add_noise(img, noise_level=0.3):
    noisy_img = img + noise_level * np.random.randn(*img.shape)
    return np.clip(noisy_img, 0, 1)

noisy_image = add_noise(image)

# 定义高斯滤波函数
def gaussian_filter(img, kernel_size=5, sigma=0.1):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    kernel /= np.sum(kernel)
    print('高斯矩阵：')
    print(kernel)
    # 对每个通道进行卷积
    filtered_img = np.zeros_like(img)
    pad_img = np.pad(img, ((kernel_size // 2,), (kernel_size // 2,), (0,)), mode='reflect')
    for c in range(img.shape[2]):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = pad_img[i:i + kernel_size, j:j + kernel_size, c]
                filtered_img[i, j, c] = np.sum(region * kernel)
    
    return filtered_img

# 定义均值滤波函数
def mean_filter(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    
    # 对每个通道进行卷积
    filtered_img = np.zeros_like(img)
    pad_img = np.pad(img, ((kernel_size // 2,), (kernel_size // 2,), (0,)), mode='reflect')
    for c in range(img.shape[2]):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = pad_img[i:i + kernel_size, j:j + kernel_size, c]
                filtered_img[i, j, c] = np.sum(region * kernel)
    
    return filtered_img

# 应用滤波器
gaussian_filtered_image = gaussian_filter(noisy_image, kernel_size=5, sigma=1.0)
mean_filtered_image = mean_filter(noisy_image, kernel_size=3)

# 取消归一化并转换为 uint8 格式
def to_uint8(img):
    return (img * 255).astype(np.uint8)

# 显示结果
plt.figure(figsize=(16, 5))
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title("原图像")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(noisy_image)
plt.title("加噪声图像")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(gaussian_filtered_image)
plt.title("高斯滤波后")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(mean_filtered_image)
plt.title("均值滤波后")
plt.axis('off')

plt.tight_layout()
plt.show()


# 保存图像
cv2.imwrite('noisy_image.jpg', cv2.cvtColor(to_uint8(noisy_image), cv2.COLOR_RGB2BGR))         # 加噪声图像
cv2.imwrite('gaussian_filtered_image.jpg', cv2.cvtColor(to_uint8(gaussian_filtered_image), cv2.COLOR_RGB2BGR))  # 高斯滤波
cv2.imwrite('mean_filtered_image.jpg', cv2.cvtColor(to_uint8(mean_filtered_image), cv2.COLOR_RGB2BGR))         # 均值滤波
