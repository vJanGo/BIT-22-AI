import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset
from gensim.models import KeyedVectors
import jieba
from transformers import BertTokenizer

class Dictionary(object):
    def __init__(self, path):
        self.word2tkn = {"[PAD]": 0, "[UNK]": 1}
        self.tkn2word = ["[PAD]", "[UNK]"]
        
        self.label2idx = {}
        self.idx2label = []

        # 获取 label 的 映射
        with open(os.path.join(path, 'labels.json'), 'r', encoding='utf-8') as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data['label'], one_data['label_desc']
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) - 1

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]


class Corpus(object):
    '''
    完成对数据集的读取和预处理，处理后得到所有文本数据的对应的 token 表示及相应的标签。
    
    该类适用于任务一、任务二，若要完成任务三，需对整个类进行调整，例如，可直接调用预训练模型提供的 tokenizer 将文本转为对应的 token 序列。
    '''
    def __init__(self, path, max_token_per_sent):
        self.dictionary = Dictionary(path)

        self.max_token_per_sent = max_token_per_sent

        # 加载Word2Vec格式的词向量
        self.word_vectors = KeyedVectors.load_word2vec_format("sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.bz2")

        self.train = self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))
        self.test = self.tokenize(os.path.join(path, 'test.json'), True)
        embedding_dim = self.word_vectors.vector_size
        self.embedding_weight = np.zeros((len(self.dictionary.tkn2word), embedding_dim))
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 若要采用预训练的 embedding, 需处理得到 token->embedding 的映射矩阵 embedding_weight。矩阵的格式参考 nn.Embedding() 中的参数 _weight
        # 注意，需考虑 [PAD] 和 [UNK] 两个特殊词向量的设置

        for idx, word in enumerate(self.dictionary.tkn2word):
            if word == "[PAD]":
                # [PAD]设置为全零
                self.embedding_weight[idx] = np.zeros(embedding_dim)
            elif word == "[UNK]":
                # [UNK]使用随机初始化
                self.embedding_weight[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
            elif word in self.word_vectors:
                self.embedding_weight[idx] = self.word_vectors[word]
            else:
                # 对于未在词向量中找到的词，使用随机初始化
                self.embedding_weight[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
        print(self.embedding_weight.shape)
        #------------------------------------------------------end------------------------------------------------------#
    def pad(self, origin_token_seq):
        '''
        padding: 将原始的 token 序列补 0 至预设的最大长度 self.max_token_per_sent
        '''
        if len(origin_token_seq) > self.max_token_per_sent:
            return origin_token_seq[:self.max_token_per_sent]
        else:
            return origin_token_seq + [0 for _ in range(self.max_token_per_sent-len(origin_token_seq))]

    def tokenize(self, path, test_mode=False):
        '''
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        '''
        idss = []
        labels = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                one_data = json.loads(line)  # 读取一条数据
                sent = one_data['sentence']
                #-----------------------------------------------------begin-----------------------------------------------------#
                # 若要采用预训练的 embedding, 需在此处对 sent 进行分词

                sent = list(jieba.cut(sent))
                #------------------------------------------------------end------------------------------------------------------#
                # 向词典中添加词
                for word in sent:
                    self.dictionary.add_word(word)

                ids = []
                for word in sent:
                    ids.append(self.dictionary.word2tkn[word])
                idss.append(self.pad(ids))
                # 测试集无标签，在 label 中存测试数据的 id，便于最终预测文件的打印
                if test_mode:
                    label = json.loads(line)['id']      
                    labels.append(label)
                else:
                    label = json.loads(line)['label']
                    labels.append(self.dictionary.label2idx[label])

            idss = torch.tensor(np.array(idss))
            
            labels = torch.tensor(np.array(labels)).long()
            
        return TensorDataset(idss, labels)
    
class BertCorpus(object):
    def __init__(self, path, max_token_per_sent):
        self.dictionary = Dictionary(path)
        self.max_token_per_sent = max_token_per_sent
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.train = self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))
        self.test = self.tokenize(os.path.join(path, 'test.json'), True)

    def pad(self, origin_token_seq):
        '''
        padding: 将原始的 token 序列补 0 至预设的最大长度 self.max_token_per_sent
        '''
        if len(origin_token_seq) > self.max_token_per_sent:
            return origin_token_seq[:self.max_token_per_sent]
        else:
            return origin_token_seq + [0 for _ in range(self.max_token_per_sent-len(origin_token_seq))]

    def tokenize(self, path, test_mode=False):
        '''
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        '''
        idss = []
        attention_masks = []
        labels = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                one_data = json.loads(line)  # 读取一条数据
                sent = one_data['sentence']
                
                # 使用BERT的tokenizer进行分词
                encoded_input = self.tokenizer(sent, padding='max_length', truncation=True, max_length=self.max_token_per_sent, return_tensors='pt')
                ids = encoded_input['input_ids'].squeeze().tolist()
                attention_mask = encoded_input['attention_mask'].squeeze().tolist()
                idss.append(ids)
                attention_masks.append(attention_mask)

                # 测试集无标签，在 label 中存测试数据的 id，便于最终预测文件的打印
                if test_mode:
                    label = json.loads(line)['id']      
                    labels.append(label)
                else:
                    label = json.loads(line)['label']
                    labels.append(self.dictionary.label2idx[label])

            idss = torch.tensor(np.array(idss))
            attention_masks = torch.tensor(np.array(attention_masks))
            labels = torch.tensor(np.array(labels)).long()
            
        return TensorDataset(idss, attention_masks, labels)
    
