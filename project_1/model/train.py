
import os
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from project_1.model.dataset import data_loader, TextDataset

PRETRAIN_MODEL_PATH = '/Users/ethan/Documents/projects/models/bert-base-chinese'


####### Step 1. 加载训练和测试数据 #######
train_file_path = "data/E-Commerce/train.txt"
val_file_path = "data/E-Commerce/dev.txt"

# 加载数据
train_lines, train_tags = data_loader(train_file_path)
val_lines, val_tags = data_loader(val_file_path)


# 统计NER标签
labels = []
for tags in train_tags: labels.extend(tags)
labels = list(set(labels))
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

train_tags = [[label2id[t] for t in tags] for tags in train_tags]
val_tags = [[label2id[t] for t in tags] for tags in val_tags]


# 数据编码
max_length = 70
tokenizer = BertTokenizer.from_pretrained(PRETRAIN_MODEL_PATH)
train_encoding = tokenizer.batch_encode_plus(train_lines, truncation=True, padding='max_length', max_length=max_length)
val_encoding = tokenizer.batch_encode_plus(val_lines, truncation=True, padding='max_length', max_length=max_length)

# 中文BERT是以字符粒度分词，校验input和tag list是否对齐
for i in range(len(train_encoding["input_ids"])):
    assert len(train_encoding["input_ids"][i]) == \
            len(train_encoding["token_type_ids"][i]) == \
            len(train_encoding["attention_mask"][i])


# 封装数据加载类
train_dataset = TextDataset(train_encoding, train_tags, max_length)
val_dataset = TextDataset(val_encoding, val_tags, max_length)

batch_size = 8  # 定义每次训练/测试的计算数据量
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
print("train sample: {}, iter_num: {}".format(len(train_lines), len(train_loader)))
print("val sample: {}, iter_num: {}".format(len(val_lines), len(val_loader)))


####### Step 2. 定义模型 #######
from project_1.model.bert import Bert

model = Bert(label2id)
epoch = 2
model.train(train_loader, epoch)





