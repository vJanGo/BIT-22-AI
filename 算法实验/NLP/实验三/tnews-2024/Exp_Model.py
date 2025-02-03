import torch.nn as nn
import torch as torch
import math
from transformers import BertModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=512, d_hid=2048, nhead=10, nlayers=6, dropout=0.2, embedding_weight=None):
        super(Transformer_model, self).__init__()
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        if embedding_weight is not None:
            embedding_weight = torch.tensor(embedding_weight, dtype=torch.float32)  # 转换为 torch.Tensor
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.pos_encoder = PositionalEncoding(d_model=d_emb, max_len=ntoken)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=d_emb, nhead=nhead, dim_feedforward=d_hid) # 要整除
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=nlayers)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 transformer 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选

        # 请自行设计分类器
        
        ## 使用了三种分类器
        ## 1. 平均池化+分类器
        ## 2. 最大池化+分类器
        ## 3. transformer decoder层+平均池化
        
        # # 使用transformer decoder层进行处理
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_emb, nhead=nhead, dim_feedforward=d_hid)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=nlayers)
        
        # 添加线性分类层
        self.classifier = nn.Sequential(
            nn.Linear(d_emb, d_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, 15) 
        )
        

        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        x = self.embed(x)     
        x = x.permute(1, 0, 2)          
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 transformer_encoder 的隐藏层输出进行处理和选择，并完成分类
        x = self.dropout(x) # 可选
        
        # 1. 平均池化
        # x = torch.mean(x, dim=1)
        
        # 2.  最大池化
        x,_ = torch.max(x, dim=1)
        
        # 3. transformer decoder层
        # 生成目标序列的掩码
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(x.device)
        # # 通过transformer decoder处理
        # x = self.transformer_decoder(x, x, tgt_mask=tgt_mask)
        # x = torch.mean(x, dim=1)
        
        
        # 通过分类器得到最终分类结果 
        x = self.classifier(x)
        #------------------------------------------------------end------------------------------------------------------#
        return x
    
    
class BiLSTM_model(nn.Module):
    def __init__(self, vocab_size, ntoken, embedding_weight,d_emb=100, d_hid=80, nlayers=1, dropout=0.2):
        super(BiLSTM_model, self).__init__()
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        if embedding_weight is not None:
            embedding_weight = torch.tensor(embedding_weight, dtype=torch.float32)  # 转换为 torch.Tensor
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.lstm = nn.LSTM(input_size=d_emb, hidden_size=d_hid, num_layers=nlayers, bidirectional=True, batch_first=True) 
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 bilstm 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选

        # 请自行设计分类器
        ## 使用同样的三种分类器
        ## 1. 平均池化+分类器
        ## 2. 最大池化+分类器
        ## 3. attention层+分类器
        
       
        ## 使用transformer decoder层进行处理
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_hid*2, nhead=8, dim_feedforward=d_hid*2)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=nlayers)
       
       
       
        self.classifier = nn.Sequential(
            nn.Linear(d_hid*2, d_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, 15) 
        )
        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        x = self.embed(x)
        x = self.lstm(x)[0]
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 bilstm 的隐藏层输出进行处理和选择，并完成分类
        x = self.dropout(x)
        
        
        # 1. 平均池化
        # x = torch.mean(x, dim=1)
        
        
        # 2. 最大池化
        # x,_ = torch.max(x, dim=1)

        
        # 3. attention层
        
        # 生成目标序列的掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(x.device)
        
        # 通过transformer decoder处理
        x = self.transformer_decoder(x, x, tgt_mask=tgt_mask)
        x = torch.mean(x, dim=1)

        # MLP
        x = self.classifier(x)

        #------------------------------------------------------end------------------------------------------------------#
        return x
    
class BertPretrainedModel(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(BertPretrainedModel, self).__init__()
        # 加载预训练的BERT模型
    
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        
        # 只微调不训练
        
        for param in self.bert.parameters():
            param.requires_grad = False


        # 使用transformer decoder层进行处理
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.bert.config.hidden_size, nhead=8, dim_feedforward=self.bert.config.hidden_size)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=2)


        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size//2, 15) 
        )
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        
        
        # 获取BERT的输出
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        x = outputs.last_hidden_state
        
        # 1. 平均池化
        # x = torch.mean(x, dim=1)
        # logits = self.classifier(x)

        # 2. 最大池化
        # x,_ = torch.max(x, dim=1)
        # logits = self.classifier(x)

        # 3. transformer decoder层
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(x.device)
        x = self.transformer_decoder(x, x, tgt_mask=tgt_mask)
        x = torch.mean(x, dim=1)
        logits = self.classifier(x)

        return logits