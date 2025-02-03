from peft import LoraConfig, get_peft_model
import pandas as pd
import configparser
import os
import torch
from transformers import AutoModelForCausalLM, ViTModel, AutoTokenizer, AutoFeatureExtractor
from model import ChangeDetectionModel
from exp_dataset import LevirDataset, create_data_loader
from tqdm import tqdm
import random
import gc

# 混合精度训练配置
from torch.cuda.amp import autocast, GradScaler
# BLEU 分数计算
from nltk.translate.bleu_score import sentence_bleu


def train(model, train_dataset, optimizer, device, epochs=3, batch_size=16, save_every=4, accumulation_steps=2):
    """
    模型训练函数
    包括混合精度训练、梯度累计、动态损失反馈和批次损失记录。
    :param model: 待训练的模型
    :param train_dataset: 训练数据集
    :param optimizer: 优化器
    :param device: 计算设备（'cpu' 或 'cuda'）
    :param epochs: 训练轮数
    :param batch_size: 每批次数据的大小
    :param save_every: 每隔指定批次数保存一次损失
    :param accumulation_steps: 梯度累计步数
    """
    model.train()
    train_loader = create_data_loader(train_dataset, batch_size=batch_size, is_train=True)
    scaler = GradScaler()  # 初始化混合精度训练的Scaler
    all_epoch_losses = []  # 存储每个 epoch 的所有 batch 损失

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        epoch_losses = []  # 当前 epoch 的 batch 损失

        with tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{epochs}", unit="batch") as pbar:
            for idx, batch in enumerate(pbar):
                image1, image2, target_ids, _, _ = batch
                image1, image2, target_ids = image1.to(device), image2.to(device), target_ids.to(device)

                # 混合精度训练
                with autocast():
                    outputs = model(image1, image2, target_ids)
                    loss = outputs.loss

                # 梯度累计
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()  # 反向传播

                if (idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)  # 更新模型参数
                    scaler.update()  # 更新 Scaler
                    optimizer.zero_grad()  # 清空梯度

                running_loss += loss.item()
                batch_count += 1
                avg_loss = running_loss / batch_count
                pbar.set_description(f"Epoch: {epoch + 1}/{epochs}, avg_loss: {avg_loss:.4f}")

                if (idx + 1) % save_every == 0:
                    epoch_losses.append(avg_loss)

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Average Loss: {avg_loss:.4f}")
        all_epoch_losses.append(epoch_losses)

    return all_epoch_losses


def evaluate_bleu(model, val_loader, device, tokenizer):
    """
    在验证集上评估模型性能，计算 BLEU 分数。
    :param model: 训练后的模型
    :param val_loader: 验证集 DataLoader
    :param device: 计算设备
    :param tokenizer: 用于解码文本的分词器
    :return: 验证集的平均 BLEU 分数
    """
    model.eval()
    total_bleu = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="validating"):
            image1, image2, target_ids, _, _ = batch
            image1, image2, target_ids = image1.to(device), image2.to(device), target_ids.to(device)

            # 文本生成
            generated_ids = model.generate(image1, image2)

            for i in range(len(image1)):
                generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                target_text = tokenizer.decode(target_ids[i], skip_special_tokens=True)

                bleu_score = sentence_bleu([target_text.split()], generated_text.split())
                total_bleu += bleu_score
                total_samples += 1

                print("Target text:", target_text)
                print("Generated text:", generated_text)

    avg_bleu = total_bleu / total_samples
    return avg_bleu


def random_sample_dataset(dataset, fraction=0.5, change_flag_threshold=0.1):
    """
    随机抽取部分数据集样本。
    :param dataset: 原始数据集
    :param fraction: 抽样比例
    :param change_flag_threshold: 对未发生变化的样本随机选择的概率
    :return: 抽样后的数据集
    """
    filtered_indices = []
    for idx in range(len(dataset)):
        change_flag = dataset[idx][4]
        if change_flag != 0 or (change_flag == 0 and random.random() < change_flag_threshold):
            filtered_indices.append(idx)

    dataset_length = len(filtered_indices)
    print(dataset_length)
    num_samples = int(dataset_length * fraction)
    sampled_indices = random.sample(filtered_indices, num_samples)
    sampled_dataset = torch.utils.data.Subset(dataset, sampled_indices)

    del dataset
    gc.collect()

    return sampled_dataset


# 定义超参数和路径
epochs = 10
batch_size = 2
learning_rate = 1e-5
device = 'cuda'
vit_hidden_size = 768
qwen_hidden_size = 1536

qwen_model_path = 'Qwen2.5-1.5B-Instruct'
vit_model_path = 'vit-base-patch16-224'
dataset_path = 'Levir-CC-dataset'

# 初始化模型
qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_path, local_files_only=True)
vit_model = ViTModel.from_pretrained(vit_model_path)
tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
feature_extractor = AutoFeatureExtractor.from_pretrained(vit_model_path)

lora_config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    lora_dropout=0.1, 
    bias="none", 
    target_modules=["q_proj", "k_proj", "v_proj"]
)
qwen_model_lora = get_peft_model(qwen_model, lora_config)
model = ChangeDetectionModel(vit_hidden_size, qwen_hidden_size, vit_model, qwen_model_lora)

# 冻结图像编码器
for param in model.vit_model.parameters():
    param.requires_grad = False

model.to(device)

# 加载数据集
train_dataset = torch.load(os.path.join(dataset_path, "train.pt"))
train_dataset = random_sample_dataset(train_dataset, 1)
val_dataset = torch.load(os.path.join(dataset_path, "valid.pt"))
val_dataset = random_sample_dataset(val_dataset, 0.3)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 训练过程
all_epoch_losses = train(model, train_dataset, optimizer, device, epochs, batch_size)
pd.DataFrame(all_epoch_losses).to_csv('train_losses.csv', index_label='Epoch')

# 合并 LoRA 层参数
model.ar_model = model.ar_model.merge_and_unload()
print(model.ar_model)
torch.save(model, "new_model.pth")

# 验证模型性能
val_loader = create_data_loader(val_dataset, batch_size=batch_size, is_train=False)
val_bleu = evaluate_bleu(model, val_loader, device, tokenizer)
print("Validation BLEU:", val_bleu)
