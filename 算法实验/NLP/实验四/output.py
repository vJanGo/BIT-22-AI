import torch
import random
from exp_dataset import create_data_loader
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

# 设置设备为 GPU
device = torch.device("cuda")

# 加载已训练的模型和分词器
model = torch.load("new_model.pth").to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-1.5B-Instruct")

# 加载测试数据集
test_dataset = torch.load("Levir-CC-dataset/test.pt")

def show_images(image1, image2, title1="Before", title2="After"):
    """
    可视化对比变化前后的两张图像。
    :param image1: 变化前的图像
    :param image2: 变化后的图像
    :param title1: 图像1的标题（默认 "Before"）
    :param title2: 图像2的标题（默认 "After"）
    """
    # 如果输入图像是张量，转换为 NumPy 数组
    if isinstance(image1, torch.Tensor):
        image1 = image1.numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.numpy()
    
    # 如果图像是 CHW 格式，转换为 HWC 格式
    if image1.shape[0] == 3:
        image1 = image1.transpose(1, 2, 0)
    if image2.shape[0] == 3:
        image2 = image2.transpose(1, 2, 0)
    
    # 创建 1 行 2 列的子图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image1)
    axes[0].set_title(title1)
    axes[0].axis('off')  # 关闭坐标轴显示
    axes[1].imshow(image2)
    axes[1].set_title(title2)
    axes[1].axis('off')  # 关闭坐标轴显示

    plt.tight_layout()
    plt.show()

def generate(model, image1, image2, max_length=20, device="cuda"):
    """
    使用已训练模型生成文本描述。
    :param model: 训练好的模型
    :param image1: 变化前的图像（张量格式）
    :param image2: 变化后的图像（张量格式）
    :param max_length: 生成文本的最大长度
    :param device: 运行设备（'cuda' 或 'cpu'）
    :return: 生成的文本 token IDs
    """
    model.eval()  # 设置模型为评估模式
    model.to(device)  # 将模型移动到指定设备

    # 添加 batch_size 维度
    image1 = image1.unsqueeze(0).to(device)
    image2 = image2.unsqueeze(0).to(device)
    
    # 提取图像特征
    with torch.no_grad():
        encoded_image1 = model.vit_model(pixel_values=image1).last_hidden_state
        encoded_image2 = model.vit_model(pixel_values=image2).last_hidden_state
        # 调整图像特征维度
        adapted_features1 = model.image_adapter(encoded_image1)
        adapted_features2 = model.image_adapter(encoded_image2)
        # 计算交叉注意力特征
        cross_attn_features1, cross_attn_features2 = model.cross_attention(adapted_features1, adapted_features2)
        # 拼接交叉注意力特征
        combined_input = torch.cat([cross_attn_features1, cross_attn_features2], dim=1)
    
    # 调整生成文本的最大长度
    max_length += combined_input.size(1)
    # 使用自回归模型生成文本
    generated_ids = model.ar_model.generate(
        inputs_embeds=combined_input,
        max_length=max_length,
        num_beams=4,  # 使用 Beam Search
        early_stopping=True,  # 提前停止生成
        pad_token_id=model.ar_model.config.pad_token_id,  # 填充 token ID
        eos_token_id=model.ar_model.config.eos_token_id,  # 结束 token ID
        do_sample=True,  # 使用随机采样
        top_k=50,  # Top-k 采样
        top_p=0.9,  # Top-p 采样
        temperature=0.7,  # 生成温度
    )
    
    return generated_ids[0]

# 从测试数据集中随机选取一条数据
one_data = test_dataset[random.randint(0, len(test_dataset) - 1)]
image1, image2, target_ids, _, _ = one_data
print(image1.shape)

# 展示变化前后图像
show_images(image1, image2)

# 解码目标文本
target_text = tokenizer.decode(target_ids, skip_special_tokens=True)

# 使用模型生成文本描述
pred_ids = generate(model, image1, image2)
pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

# 打印目标文本和生成的文本
print("Target text:", target_text)
print("Predict text:", pred_text)
