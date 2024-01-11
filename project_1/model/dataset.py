import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, encodings, labels, maxlen):
        self.encodings = encodings
        self.labels = labels
        self.maxlen = maxlen

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx][:self.maxlen]) for key, value in self.encodings.items()}
        # 字级别的标注，注意填充cls，这里[0]代表cls。后面不够长的这里也是补充0，样本tokenizer的时候已经填充了
        item['labels'] = torch.tensor([0] + self.labels[idx] + [0] * (self.maxlen - 1 - len(self.labels[idx])))[:self.maxlen]
        return item

    def __len__(self):
        return len(self.labels)


def data_loader(file_path):
    sentence, lines, tags = [], [], []
    for line in open(file_path).readlines():
        if len(line.rstrip()) == 0:
            lines.append([s.split("\t")[0] for s in sentence])
            tags.append([s.split("\t")[1] for s in sentence])
            assert len(lines[-1]) == len(tags[-1])
            sentence = []
        else:
            sentence.append(line.rstrip())
    return lines, tags

