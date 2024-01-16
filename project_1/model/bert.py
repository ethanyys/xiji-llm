
import torch
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup

PRETRAIN_MODEL_PATH = '/Users/ethan/Documents/projects/models/bert-base-chinese'

class Bert:
    def __init__(self, label2id):
        self.model = BertForTokenClassification.from_pretrained(PRETRAIN_MODEL_PATH, num_labels=len(label2id))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

    def train(self, train_loader, epoch):
        total_steps = len(train_loader) * epoch
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        self.model.train()
        iter_num = 0
        total_iter = len(train_loader) * epoch      # 训练过程中的总体计算次数
        for e in range(epoch):
            total_train_loss = 0
            for idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]   # batch loss
                # loss = outputs.loss
                logits = outputs[1]  # shape: [batch, max_len, label_num]
                out = logits.argmax(dim=2).data

                # if idx % 20 == 0:  # 看模型的准确率(当前batch)
                #     with torch.no_grad():
                #         print((out == labels.data).float().mean().item(), loss.item())

                total_train_loss += loss.item()

                # 反向梯度信息
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # 参数更新
                self.optimizer.step()
                scheduler.step()

                iter_num += 1
                if iter_num % 20 == 0:
                    print("Epoch: {}, iter_num: {}, loss: {}, total_progress: {}".format(
                        e, iter_num, loss.item(), iter_num / total_iter))

            print("Epoch: {}, Average training loss: {}".format(e, total_train_loss / len(train_loader)))
