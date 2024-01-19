
import torch
import numpy as np
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup

from project_1.model.metric import metric


PRETRAIN_MODEL_PATH = '/Users/ethan/Documents/projects/models/bert-base-chinese'


class Bert:
    def __init__(self, label2id, maxlen):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.label2id = label2id
        self.max_len = maxlen

    def init_model(self, model_path):
        self.model = BertForTokenClassification.from_pretrained(model_path, num_labels=len(self.label2id))
        self.model.to(self.device)

    def load_ft_model(self, model_path):
        self.model = torch.load(model_path, map_location=self.device)

    def train(self, train_loader, epoch):
        total_steps = len(train_loader) * epoch
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        self.model.train()
        iter_num = 0
        total_iter = len(train_loader) * epoch      # 训练过程中的总体计算次数
        for e in range(epoch):
            total_train_loss = 0
            for idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]   # batch loss
                logits = outputs[1]  # shape: [batch, max_len, label_num]
                # out = logits.argmax(dim=2).data

                total_train_loss += loss.item()

                # 反向梯度信息
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # 参数更新
                optimizer.step()
                scheduler.step()

                iter_num += 1
                if iter_num % 20 == 0:
                    print("Epoch: {}, iter_num: {}, loss: {}, total_progress: {}".format(
                        e, iter_num, loss.item(), iter_num / total_iter))

            print("Epoch: {}, Average training loss: {}".format(e, total_train_loss / len(train_loader)))

    def valid(self, valid_loader):
        self.model.eval()
        total_eval_accuracy, total_eval_loss = 0, 0
        preds, truths = [], []
        for batch in valid_loader:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs[1]

            total_eval_loss += loss.item()
            label_ids = labels.to('cpu').numpy()
            pred_ids = logits.argmax(dim=2).data.to('cpu').numpy()  # shape = (batch, max_len)
            preds.append(pred_ids)
            truths.append(label_ids)

            total_eval_accuracy += (outputs[1].argmax(2).data == labels.data).float().mean().item()

        # 计算 precision、recall、f1-score
        preds = np.concatenate(preds, axis=0)
        truths = np.concatenate(truths, axis=0)

        tot_precision, tot_recall, tot_f1_score, \
            precision, recall, f1_score = metric(truths, preds, self.label2id)

        avg_val_accuracy = total_eval_accuracy / len(valid_loader)
        print("Accuracy: %.4f" % (avg_val_accuracy))
        print("Average testing loss: %.4f" % (total_eval_loss / len(valid_loader)))
        print("-------------------------------\n")

        print("Total Precision: {}, Recall: {}, F1-Score: {}".format(tot_precision, tot_recall, tot_f1_score))
        for label in f1_score:
            print("{} Precision: {}, Recall: {}, F1-Score: {}".format(label, precision[label], recall[label], f1_score[label]))

    def predict(self, input_sentence, tokenizer):
        input_sentence = tokenizer([input_sentence],
                          truncation=True,
                          padding='max_length',
                          max_length=64)

        with torch.no_grad():
            input_ids = torch.tensor(input_sentence['input_ids']).to(self.device).reshape(1, -1)
            attention_mask = torch.tensor(input_sentence['attention_mask']).to(self.device).reshape(1, -1)
            labels = torch.tensor([0] * attention_mask.shape[1]).to(self.device).reshape(1, -1)

            outputs = self.model(input_ids, attention_mask, labels)
            outputs = outputs[0].data.cpu().numpy()  # (1, max_len, num_label)

            outputs = outputs[0].argmax(1)[1:-1]
