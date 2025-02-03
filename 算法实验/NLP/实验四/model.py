import torch
import torch.nn as nn

class Project(nn.Module):
    """
    用于将图像编码器的隐藏特征维度映射到自回归模型输入的隐藏特征维度。
    """
    def __init__(self, vit_hidden_size, ar_hidden_size):
        """
        初始化投影层。
        :param vit_hidden_size: ViT 模型的隐藏层维度大小
        :param ar_hidden_size: 自回归生成模型的隐藏层维度大小
        """
        super().__init__()
        self.linear_projection = nn.Linear(vit_hidden_size, ar_hidden_size)

    def forward(self, image_features):
        """
        前向传播，将图像特征通过线性变换映射到目标维度。
        :param image_features: 输入图像特征
        :return: 映射后的图像特征
        """
        return self.linear_projection(image_features)


class CrossAttention(nn.Module):
    """
    Cross Attention 模块，用于对齐两幅图像的信息。
    """
    def __init__(self, hidden_size):
        """
        初始化 Cross Attention 模块。
        :param hidden_size: 输入特征的隐藏层维度大小
        """
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)

    def forward(self, tokens1, tokens2):
        """
        前向传播，计算两幅图像特征之间的交叉注意力。
        :param tokens1: 第一幅图像的特征
        :param tokens2: 第二幅图像的特征
        :return: 两幅图像经过交叉注意力后的特征
        """
        attn_output1, _ = self.multihead_attention(tokens1, tokens2, tokens2)
        attn_output2, _ = self.multihead_attention(tokens2, tokens1, tokens1)
        return attn_output1, attn_output2


class ChangeDetectionModel(nn.Module):
    """
    遥感图像变化检测模型，结合图像编码器和自回归模型完成变化描述生成任务。
    """
    def __init__(self, vit_hidden_size, ar_hidden_size, vit_model, ar_model):
        """
        初始化模型。
        :param vit_hidden_size: 图像编码器（ViT）的隐藏层维度大小
        :param ar_hidden_size: 自回归生成模型的隐藏层维度大小
        :param vit_model: 图像编码器（ViT模型实例）
        :param ar_model: 自回归生成模型（大语言模型实例）
        """
        super().__init__()
        self.image_encoder = vit_model  # 图像编码器
        self.image_adapter = Project(vit_hidden_size, ar_hidden_size)  # 图像特征适配器
        self.cross_attention = CrossAttention(hidden_size=ar_hidden_size)  # 交叉注意力模块
        self.text_generator = ar_model  # 自回归生成模型

    def forward(self, image_before, image_after, target_tokens):
        """
        模型的前向传播，用于训练阶段。
        :param image_before: 变化前的图像
        :param image_after: 变化后的图像
        :param target_tokens: 目标文本的 token 编码
        :return: 模型的输出，包含损失值
        """
        # 提取图像特征
        encoded_image_before = self.image_encoder(pixel_values=image_before).last_hidden_state[:, 0:1, :]
        encoded_image_after = self.image_encoder(pixel_values=image_after).last_hidden_state[:, 0:1, :]

        # 适配特征维度
        adapted_features_before = self.image_adapter(encoded_image_before)
        adapted_features_after = self.image_adapter(encoded_image_after)

        # 交叉注意力对齐
        aligned_features_before, aligned_features_after = self.cross_attention(adapted_features_before, adapted_features_after)

        # 合并特征
        combined_image_features = torch.cat([aligned_features_before, aligned_features_after], dim=1)

        # 获取目标文本的嵌入表示
        target_embeddings = self.text_generator.get_input_embeddings()(target_tokens)

        # 将图像特征与目标文本嵌入拼接
        input_embeddings = torch.cat([combined_image_features, target_embeddings], dim=1)

        # 为图像特征部分的 token 标记 -100（不计算损失）
        labels = torch.cat([
            torch.full((target_tokens.size(0), combined_image_features.size(1)), -100).to(target_tokens.device),
            target_tokens
        ], dim=1)

        # 自回归生成模型的前向传播
        outputs = self.text_generator(inputs_embeds=input_embeddings, labels=labels)

        return outputs

    def generate(self, image_before, image_after, max_length=30):
        """
        模型的文本生成，用于推理阶段。
        :param image_before: 变化前的图像
        :param image_after: 变化后的图像
        :param max_length: 生成文本的最大长度
        :return: 生成的文本 token IDs
        """
        # 提取图像特征
        with torch.no_grad():
            encoded_image_before = self.image_encoder(pixel_values=image_before).last_hidden_state[:, 0:1, :]
            encoded_image_after = self.image_encoder(pixel_values=image_after).last_hidden_state[:, 0:1, :]
            adapted_features_before = self.image_adapter(encoded_image_before)
            adapted_features_after = self.image_adapter(encoded_image_after)
            aligned_features_before, aligned_features_after = self.cross_attention(adapted_features_before, adapted_features_after)
            
            # 计算变化特征（差异特征）
            diff_features = torch.abs(aligned_features_before - aligned_features_after)
            combined_image_features = torch.cat([aligned_features_before, aligned_features_after, diff_features], dim=1)

        # 调整生成长度
        max_length += combined_image_features.size(1)

        # 调用自回归模型生成文本
        generated_ids = self.text_generator.generate(
            inputs_embeds=combined_image_features, 
            max_length=max_length,
            num_beams=4, 
            early_stopping=True, 
            pad_token_id=self.text_generator.config.pad_token_id, 
            eos_token_id=self.text_generator.config.eos_token_id
        )
        
        return generated_ids

