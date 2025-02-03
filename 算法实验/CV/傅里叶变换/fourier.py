import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'

def display_image(title, image, cmap='gray'):
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()

def dft2d(image):
    """计算二维离散傅里叶变换"""
    rows, cols = image.shape
    transform = np.zeros((rows, cols), dtype=complex)
    for u in range(rows):
        for v in range(cols):
            sum_value = 0.0
            for x in range(rows):
                for y in range(cols):
                    exponent = -2j * np.pi * ((u * x / rows) + (v * y / cols))
                    sum_value += image[x, y] * np.exp(exponent)
            transform[u, v] = sum_value
    return transform

def idft2d(f_transform):
    """计算二维逆离散傅里叶变换"""
    rows, cols = f_transform.shape
    image = np.zeros((rows, cols), dtype=complex)
    for x in range(rows):
        for y in range(cols):
            sum_value = 0.0
            for u in range(rows):
                for v in range(cols):
                    exponent = 2j * np.pi * ((u * x / rows) + (v * y / cols))
                    sum_value += f_transform[u, v] * np.exp(exponent)
            image[x, y] = sum_value / (rows * cols)
    return np.abs(image)

def apply_mask(f_transform, mask):
    """应用频率掩膜"""
    return f_transform * mask

# 1. 创建原始图像
image = np.zeros((64, 64), dtype=float)
image[24:40, 24:40] = 255  # 创建一个白色正方形
display_image("原始图像", image)


# 2. 计算傅里叶变换
f_transform = dft2d(image)

# 转移到中心点并显示频谱图像
f_shift = np.fft.fftshift(f_transform)
magnitude_spectrum = np.log(np.abs(f_shift) + 1)
display_image("频谱图像", magnitude_spectrum)

# 3. 去除高频部分
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# 定义低通滤波器掩膜
radius = 15
low_pass_mask = np.zeros((rows, cols), dtype=float)
for i in range(rows):
    for j in range(cols):
        if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= radius:
            low_pass_mask[i, j] = 1

# 应用低通掩膜
low_pass_transform = apply_mask(f_shift, low_pass_mask)
low_pass_transform = np.fft.ifftshift(low_pass_transform)  # 返回原点位置
low_pass_image = idft2d(low_pass_transform)
display_image("低通滤波后的图像", low_pass_image)

# 4. 去除低频部分
high_pass_mask = 1 - low_pass_mask  # 高通滤波掩膜

# 应用高通掩膜
high_pass_transform = apply_mask(f_shift, high_pass_mask)
high_pass_transform = np.fft.ifftshift(high_pass_transform)  # 返回原点位置
high_pass_image = idft2d(high_pass_transform)
display_image("高通滤波后的图像", high_pass_image)
