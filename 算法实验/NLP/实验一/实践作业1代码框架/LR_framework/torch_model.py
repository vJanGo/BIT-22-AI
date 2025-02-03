import numpy as np
import random
import argparse
import torch
import torch.nn as nn
from data_preprocess import *
from matplotlib import pyplot as plt
np.random.seed(1120221586)
"""
逻辑回归模型
请在pass处按注释要求插入代码以完善模型功能
"""

class logistic_net(nn.Module):
    """
    logistic network.
    """
    def __init__(self):
        super(logistic_net, self).__init__()
        self.layer = nn.Sequential(
                    nn.Linear(24000,1),
                    nn.Sigmoid()
        )

    def forward(self, x):
        
        x = self.layer(x)
        return x
    
def train(model,train_iter,loss_func, optimizer):
    model.train()
    total_loss = 0
    for i,data in enumerate(train_iter):
        optimizer.zero_grad()  # 清除上一步的梯度
        input = data[0]
        label = data[1]
    
        output = model(input)
        label = label.unsqueeze(1).float()
        loss = loss_func(output,label)
        
        total_loss += loss.item()
        
        # 反向传播并优化
        loss.backward()
        optimizer.step()
    
    return total_loss
    
    
def test(test_iter,model,loss_func):    
    total_loss = 0
    n_samples = len(test_set)
    for i,data in enumerate(test_iter):
        input = data[0]
        label = data[1]
    
        output = model(input)
        label = label.unsqueeze(1).float()
        loss = loss_func(output,label)
        
        total_loss += loss.item()
        
    
    return total_loss/n_samples

def test_accuracy(test_set,model):
        """
        测试模型分类精度
        :param test_set: 测试集
        :return: 模型精度（百分数）
        """
        rights = 0  # 分类正确的个数
        n_samples = len(test_set)
        for data, label in test_set:
            """
            记录分类正确的样本个数
            """
            # 进行正向传播获取模型输出
            data = torch.tensor(data, dtype=torch.float32)
            output = model(data)
            predictions = (output >= 0.5).float()  # 将输出转换为 0 或 1 的 Tensor

            # 记录分类正确的样本个数
            rights += (predictions.view(-1) == label).sum().item()  # 确保形状一致
            
        print(rights,n_samples)
        return (rights / n_samples) * 100

def draw_loss(train_losses, test_losses):
    """
    绘制损失曲线图
    :param train_losses: 训练损失
    :param test_losses: 测试损失
    :return:
    """
    epochs = range(1, len(train_losses)+1)
    plt.plot(epochs, train_losses, 'r', label='Train Loss')
    plt.plot(epochs, test_losses, 'b', label='Test Loss')
    plt.legend()

    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
if __name__ == "__main__":
    
    train_set_path = 'Dataset/train_set.pkl'
    test_set_path = 'Dataset/test_set.pkl'
    BATCH_SIZE = 16
    EPOCHS = 500
    lr = 1e-4 # 学习率
    best_loss = float('inf')  # 初始化最优损失
    train_losses,test_losses = [],[]
    # 加载训练集：
    with open(train_set_path, 'rb') as f:
        train_set = pickle.load(f)

    # 加载测试集：
    with open(test_set_path, 'rb') as f:
        test_set = pickle.load(f)
    # 制作迭代器
    train_iter = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_iter = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    # 创建模型实例
    model = logistic_net()
    loss_func = nn.BCELoss()
    optimizer= torch.optim.SGD(model.parameters(),lr=lr)
    for epoch in range(EPOCHS):
        
        train_loss = 0
        n_samples = len(train_set)
        train_loss = train(model,train_iter,loss_func,optimizer)
        train_loss /= n_samples
        test_loss = test(test_iter,model,loss_func)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        a = test_accuracy(test_set,model)
        print("Epoch:{}/{} training loss:{}, test loss:{}".format(epoch+1, EPOCHS, train_loss, test_loss))
        print("test_acc:{}".format(a))
        
        # 检查是否为最优模型
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'Model/torch_LR/best_model.pth')  # 保存模型参数
        
    # 绘制损失曲线
    draw_loss(train_losses, test_losses)