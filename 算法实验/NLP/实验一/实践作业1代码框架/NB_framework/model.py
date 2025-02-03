# -*- coding: utf-8 -*-
import jieba
import numpy as np
from collections import Counter
import time  # 用于计时
"""
Naive Bayes句子分类模型
"""

train_path = "./Dataset_NB/train.txt"
test_path = "./Dataset_NB/test.txt"

sum_words_neg = 0   # 训练集负向语料的总词数
sum_words_pos = 0   # 训练集正向语料的总词数

neg_sents_train = []  # 训练集中负向句子
pos_sents_train = []  # 训练集中正向句子
neg_sents_test = []  # 测试集中负向句子
pos_sents_test = []  # 测试集中正向句子
stopwords = []  # 停用词

def mystrip(ls):
    """
    消除句尾换行
    """
    for i in range(len(ls)):
        ls[i] = ls[i].strip("\n")
    return ls

def remove_stopwords(_words):
    """
    去掉停用词
    :param _words: 分词后的单词list
    :return: 去除停用词（无意义词）后的list
    """
    _i = 0
    for _ in range(len(_words)):
        if _words[_i] in stopwords:
            _words.pop(_i)
        else:
            _i += 1
    return _words

def my_init():
    """
    函数功能：对训练集做统计，记录训练集中正向和负向的单词数，并记录正向或负向条件下，每个词的出现次数，并收集测试句子
    return: 负向词频表，正向词频表
    """
    neg_words = []  # 负向词列表
    _neg_dict = {}  # 负向词频表
    pos_words = []  # 正向词列表
    _pos_dict = {}  # 正向词频表

    global sum_words_neg, sum_words_pos, neg_sents_train, pos_sents_train, stopwords

    # 读入stopwords
    with open("./参考资源/stopwords.txt", encoding="utf-8") as f:
        stopwords = f.readlines()
        stopwords = mystrip(stopwords)

    # 收集训练集正、负向的句子
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip('\n')
            if line[0] == "0":
                neg_sents_train.append(line[1:])
            else:
                pos_sents_train.append(line[1:])

    # 收集测试集正、负向的句子
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip('\n')
            if line[0] == "0":
                neg_sents_test.append(line[1:])
            else:
                pos_sents_test.append(line[1:])

    # 获得负向训练语料的词列表neg_words
    for i in range(len(neg_sents_train)):
        words = []
        words = jieba.lcut(neg_sents_train[i], cut_all=True)
        neg_words.append(remove_stopwords(words))
    
    # 获得正向训练语料的词列表pos_words
    for i in range(len(pos_sents_train)):
        words = []
        words = jieba.lcut(pos_sents_train[i], cut_all=True)
        pos_words.append(remove_stopwords(words))
    
    # 获得负向训练语料的词频表_neg_dict  
    for sentence in neg_words:
        for word in sentence:
            sum_words_neg += 1
            if word in _neg_dict:
                _neg_dict[word] += 1
            else:
                _neg_dict[word] = 1
    
    # 获得正向训练语料的词频表_pos_dict  
    for sentence in pos_words:
        for word in sentence:
            sum_words_pos += 1
            if word in _pos_dict:
                _pos_dict[word] += 1
            else:
                _pos_dict[word] = 1

    # 计算TF-IDF
    neg_sens_len = len(neg_words)
    pos_sens_len = len(pos_words)
    total_sens_len = neg_sens_len + pos_sens_len
    
    for word, count in _neg_dict.items():
        tf = count / sum_words_neg
        a_df = sum(1 for sentence in neg_words if word in sentence)  # 包含该词的文档数
        b_df = sum(1 for sentence in pos_words if word in sentence)
        idf = np.log(total_sens_len / (a_df + b_df + 1))  # 防止除0
        tfidf = tf * idf
        _neg_dict[word] = tfidf
        
    for word, count in _pos_dict.items():
        tf = count / sum_words_pos
        a_df = sum(1 for sentence in neg_words if word in sentence)  # 包含该词的文档数
        b_df = sum(1 for sentence in pos_words if word in sentence)
        idf = np.log(total_sens_len / (a_df + b_df + 1))  # 防止除0
        tfidf = tf * idf
        _pos_dict[word] = tfidf
    
    return _neg_dict, _pos_dict

def calculate_class(sent, neg_dict, pos_dict):
    pos_score = 0
    neg_score = 0
    for word in sent:
        if word in pos_dict and word != ' ':
            pos_score += np.log(pos_dict[word])
        else:
            pos_score += np.log(1e-8)  # 平滑处理
    for word in sent:
        if word in neg_dict and word != '':
            neg_score += np.log(neg_dict[word])
        else:
            neg_score += np.log(1e-8)
            
    return pos_score - neg_score

if __name__ == "__main__":
    start_time = time.time()  # 开始计时

    # 统计训练集：
    neg_dict, pos_dict = my_init()
    
    end_time = time.time()  # 结束计时
    # 测试：
    TP, FN, FP, TN = 0, 0, 0, 0
    
    for i in range(len(neg_sents_test)):  # 用negative的句子做测试
        st = jieba.lcut(neg_sents_test[i])  # 分词，返回词列表
        st = remove_stopwords(st)  # 去掉停用词

        score = calculate_class(st, neg_dict, pos_dict)
        if score < 0:
            TN += 1
        else:
            FP += 1

    for i in range(len(pos_sents_test)):  # 用positive的数据做测试
        st = jieba.lcut(pos_sents_test[i])
        st = remove_stopwords(st)

        score = calculate_class(st, neg_dict, pos_dict)
        if score > 0:
            TP += 1
        else:
            FN += 1

    

    acc = (TP + TN) / (len(pos_sents_test) + len(neg_sents_test)) * 100
    error = (FN + FP) / (len(pos_sents_test) + len(neg_sents_test)) * 100
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
    print(f"准确率: {acc:.1f}%, 错误率: {error:.1f}%, 精准率: {precision:.1f}%, 召回率: {recall:.1f}%, F1: {F_score:.1f}")
    print(f"训练时间: {end_time - start_time:.2f} 秒")
