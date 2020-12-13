import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from EasyTransformer import transformer
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.Encoder = transformer.TransformerEncoder(30000)
        self.linear=nn.Linear(512,49)

    def forward(self, src):
        word_vec,sent_vec=self.Encoder(src)
        return self.linear(sent_vec)

