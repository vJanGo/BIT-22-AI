import os
import shutil
import random

def split_data(source_images_dir, source_labels_dir, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, split_ratio=0.8):
    # 获取所有图像文件
    images = [f for f in os.listdir(source_images_dir) if os.path.isfile(os.path.join(source_images_dir, f))]
    print(len(images))
    # 打乱图像文件列表
    random.shuffle(images)
    
    # 计算训练集和验证集的分界点
    split_index = int(len(images) * split_ratio)
    
    # 分割图像文件为训练集和验证集
    train_images = images
    print(len(train_images))
    val_images = images[split_index:]
    print(len(val_images))
    
    # 确保目标目录存在
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # 复制训练集图像和标签
    for image in train_images:
        shutil.copy(os.path.join(source_images_dir, image), os.path.join(train_images_dir, image))
        label = image.replace('.png', '.txt')  
        shutil.copy(os.path.join(source_labels_dir, label), os.path.join(train_labels_dir, label))
    
    # 复制验证集图像和标签
    for image in val_images:
        shutil.copy(os.path.join(source_images_dir, image), os.path.join(val_images_dir, image))
        label = image.replace('.png', '.txt')  
        shutil.copy(os.path.join(source_labels_dir, label), os.path.join(val_labels_dir, label))
    print('finish')
# 调用函数
split_data('weeddetection/train/images', 
           'weeddetection/train/yolo_labels', 
           'weeddetection/datasets/train/images', 
           'weeddetection/datasets/train/labels', 
           'weeddetection/datasets/val/images', 
           'weeddetection/datasets/val/labels')
