from transformers import BertModel
import torch.nn as nn
class BERT_Classifier(nn.Module):
    def __init__(self,label_num):
        super().__init__()
        self.encoder = BertModel.from_pretrained('./bert-base-chinese')
        self.dropout = nn.Dropout(0.1,inplace=False)
        self.fc = nn.Linear(768, label_num)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x, mask,label):
        x = self.encoder(x, attention_mask=mask)[0]
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.fc(x)

        return self.criterion(x,label),x
