import json
import os
import random

from transformers import AutoTokenizer, AutoFeatureExtractor
from torch.utils.data import TensorDataset, DataLoader
import torch
from PIL import Image
import time


class LevirDataset(object):
    """
    用于解析和载入 Levir 数据集
    """
    def __init__(self, dataset_path: str, tokenizer, feature_extractor, max_token: int = 30, load_dir=None, save_dir=None):
        """
        数据集类初始化
        :param dataset_path: 数据集的根目录
        :param tokenizer: transformers库中的AutoTokenizer
        :param feature_extractor: transformers库中 ViT 模型的特征提取器， 用于预处理图像数据
        :param max_token: 最大token数
        :param load_dir: 用于载入数据集的 directory
        :param save_dir: 用与保存数据集的 directory, 一般在预处理完后保存，方便下次直接载入
        """
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_token = max_token
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.train = None
        self.valid = None
        self.test = None

        # 构建训练集、验证集、测试集
        if load_dir is None:
            self.train, self.valid, self.test = self.tokenize()   # 需要预处理 和 tokenize
        else:
            self.train = torch.load(os.path.join(self.load_dir, "train.pt"))
            self.valid = torch.load(os.path.join(self.load_dir, "valid.pt"))
            self.test = torch.load(os.path.join(self.load_dir, "test.pt"))
            print(f"{self.load_dir}: 数据集载入成功")

        if save_dir is not None:
            torch.save(self.train, os.path.join(self.save_dir, "train.pt"))
            torch.save(self.valid, os.path.join(self.save_dir, "valid.pt"))
            torch.save(self.test, os.path.join(self.save_dir, "test.pt"))
            print(f"数据集已保存在 {self.save_dir}")



    def tokenize(self) -> (TensorDataset, TensorDataset, TensorDataset):
        """
        根据 Levir 数据集下的 json 文件，解析数据集并且把数据包装成 TensorDataset
        :return: 训练集、验证集、测试集的 TensorDataset
        """
        json_file_path = os.path.join(self.dataset_path, "LevirCCcaptions.json")
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            data = data["images"]   # 现在的 data 是一个列表，每一项都是一组数据（两幅图 + text）
        
        # 获取 train, valid, test 数据
        train_data = [entry for entry in data if entry['split'] == 'train']
        valid_data = [entry for entry in data if entry['split'] == 'val']
        test_data = [entry for entry in data if entry['split'] == 'test']

        # 对训练集、验证集和测试集进行处理
        train_before_images, train_after_images = self.preprocess_image_data(
            train_data,
            os.path.join(self.dataset_path, "images", "train")
        )
        valid_before_images, valid_after_images = self.preprocess_image_data(
            valid_data,
            os.path.join(self.dataset_path, "images", "val")
        )
        test_before_images, test_after_images = self.preprocess_image_data(
            test_data,
            os.path.join(self.dataset_path, "images", "test")
        )
        
        train_target_ids, train_attention_masks, train_change_flags = self.preprocess_text_data(train_data)
        valid_target_ids, valid_attention_masks, valid_change_flags = self.preprocess_text_data(valid_data)
        test_target_ids, test_attention_masks, test_change_flags = self.preprocess_text_data(test_data)
        
        # 包装成 TensorDataset
        train_dataset = TensorDataset(train_before_images, train_after_images, train_target_ids, train_attention_masks, train_change_flags)
        valid_dataset = TensorDataset(valid_before_images, valid_after_images, valid_target_ids, valid_attention_masks, valid_change_flags)
        test_dataset =  TensorDataset(test_before_images, test_after_images, test_target_ids, test_attention_masks, test_change_flags)
        
        return train_dataset, valid_dataset, test_dataset


    def preprocess_text_data(self, data):
        """
        预处理 数据集文本部分
        :param data: list， 每一条是一组数据
        :return:
        """
        target_ids = []
        attention_masks = []
        change_flags = []
        for entry in data:
            sentence = entry['sentences'][random.randint(0, 4)]   # 随机选一句话
            change_flags.append(entry['changeflag'])

            raw = sentence['raw']
            encoding = self.tokenizer.encode_plus(raw,    # 原句子
                                      truncation=True,    # 截断
                                      padding='max_length',  # 填充策略
                                      max_length=self.max_token,
                                      return_tensors='pt' # 返回 pytorch tensor
            )
            # 此处需要去掉多余的 batch 维度
            target_ids.append(encoding['input_ids'].squeeze(0))
            attention_masks.append(encoding['attention_mask'].squeeze(0))
        return torch.stack(target_ids), torch.stack(attention_masks), torch.tensor(change_flags).long()

    def preprocess_image_data(self, data, image_dir):
        """
        预处理 数据集图像部分
        :param data: list， 每一条是一组数据
        :param image_dir: 当前data对应的image的根目录
        :return:
        """
        before_images = []
        after_images = []
        image1_dir = os.path.join(image_dir, "A")
        image2_dir = os.path.join(image_dir, "B")
        for entry in data:
            print(6)
            filename = entry['filename']
            image1 = Image.open(os.path.join(image1_dir, filename))
            image2 = Image.open(os.path.join(image2_dir, filename))
            before_image = self.feature_extractor(
                images = image1,
                return_tensors = "pt"
            )
            after_image = self.feature_extractor(
                images = image2,
                return_tensors = "pt"
            )
            # 需要手动去掉 batch_size 维度
            before_images.append(before_image["pixel_values"].squeeze(0))
            after_images.append(after_image["pixel_values"].squeeze(0))

        return torch.stack(before_images), torch.stack(after_images)






def create_data_loader(dataset: TensorDataset, batch_size: int, is_train=False):
    """
    根据 TensorDataset 创建 DataLoader
    :param dataset: 数据集
    :param batch_size: 批处理大小
    :param is_train: 是否是训练集
    :return: DataLoader 对象
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


if __name__ == '__main__':
    # 设置 tokenizer 和数据路径
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-1.5B-Instruct")
    feature_extractor = AutoFeatureExtractor.from_pretrained("vit-base-patch16-224")
    dataset_path = "Levir-CC-dataset"
    max_token = 30
    
    # 创建 LevirDataset 对象
    s_t = time.perf_counter()
    levir_dataset = LevirDataset(dataset_path, tokenizer, feature_extractor, max_token, save_dir=dataset_path)
    e_t = time.perf_counter()
    print(f"数据集构架时间： {e_t-s_t:.3f} seconds")

    # 创建 DataLoader
    train_loader = create_data_loader(levir_dataset.train, batch_size=32)
    valid_loader = create_data_loader(levir_dataset.valid, batch_size=32)
    test_loader = create_data_loader(levir_dataset.test, batch_size=32)

    # 使用数据加载器进行训练等操作
    for batch in train_loader:
        before_images, after_images, target_ids, attention_masks, change_flags = batch
        print(before_images.shape, after_images.shape, target_ids.shape, attention_masks.shape, change_flags.shape)
        break
