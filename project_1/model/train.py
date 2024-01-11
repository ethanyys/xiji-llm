
import os
from transformers import BertTokenizer

from project_1.model.dataset import data_loader

################## 训练NER模型 ##################


####### Step 1. 加载训练和测试数据 #######
train_file_path = "data/E-Commerce/train.txt"
test_file_path = "data/E-Commerce/test.txt"

# 加载数据
train_lines, train_tags = data_loader(train_file_path)
test_lines, test_tags = data_loader(test_file_path)
print(len(train_lines[0]))

# 统计NER标签
labels = []
for tags in train_tags: labels.extend(tags)
labels = list(set(labels))
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

train_tags = [[label2id[t] for t in tags] for tags in train_tags]
test_tags = [[label2id[t] for t in tags] for tags in test_tags]

# 数据编码
max_length = 70
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
train_encoding = tokenizer.batch_encode_plus(list(train_lines), truncation=True, padding=True, max_length=max_length)
val_encoding = tokenizer.batch_encode_plus(list(test_lines), truncation=True, padding=True, max_length=max_length)
print(len(train_encoding[0]))

