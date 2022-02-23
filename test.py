import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings
import torch
import time
import argparse
import json
import os
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import BertConfig
#from transformers import BertPreTrainedModel
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

setup_seed(44)

from transformers import BertModel

from loader import map_id_rel

rel2id, id2rel = map_id_rel()

print(len(rel2id))
print(id2rel)

USE_CUDA = torch.cuda.is_available()

def get_train_args():
    labels_num=len(rel2id)
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=1,help = '每批数据的数量')
    parser.add_argument('--nepoch',type=int,default=30,help = '训练的轮次')
    parser.add_argument('--lr',type=float,default=0.001,help = '学习率')
    parser.add_argument('--gpu',type=bool,default=True,help = '是否使用gpu')
    parser.add_argument('--num_workers',type=int,default=2,help='dataloader使用的线程数量')
    parser.add_argument('--num_labels',type=int,default=len(id2rel),help='分类类数')
    parser.add_argument('--data_path',type=str,default='./data',help='数据路径')
    opt=parser.parse_args()
    print(opt)
    return opt

def get_model(opt):
    model = BertForSequenceClassification.from_pretrained('./bert-base-chinese',num_labels=opt.num_labels)
    return model


def test(net,text_list,ent1_list,ent2_list,result):
    net.eval()
    max_length=128
    
    net=torch.load('model.pth')
    rel_list=[]
    with torch.no_grad():
        for text,ent1,ent2,label in zip(text_list,ent1_list,ent2_list,result):
            sent = ent1 + ent2+ text
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
            avai_len = len(indexed_tokens)
            while len(indexed_tokens) < max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

            # Attention mask
            att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            if USE_CUDA:
                indexed_tokens = indexed_tokens.cuda()
                att_mask = att_mask.cuda()

            if USE_CUDA:
                indexed_tokens=indexed_tokens.cuda()
                att_mask=att_mask.cuda()
            outputs = net(indexed_tokens, attention_mask=att_mask)
            # print(y)
            logits = outputs[0]
            _, predicted = torch.max(logits.data, 1)
            result=predicted.cpu().numpy().tolist()[0]
            print("Source Text: ",text)
            print("Entity1: ",ent1," Entity2: ",ent2," Predict Relation: ",id2rel[result]," True Relation: ",label)
            print('\n')
            rel_list.append(id2rel[result])
    return rel_list
opt = get_train_args()
model=get_model(opt)

if USE_CUDA:
    model=model.cuda()

from random import choice

text_list=[]
ent1=[]
ent2=[]
result=[]
with open("train.json", 'r', encoding='utf-8') as load_f:
    lines=load_f.readlines()
    total_num=10
    while total_num>0:
        line=choice(lines)
        dic = json.loads(line)
        text_list.append(dic['text'])
        ent1.append(dic['ent1'])
        ent2.append(dic['ent2'])
        result.append(dic['rel'])
        total_num-=1
        if total_num<0:
            break

test(model,text_list,ent1,ent2,result)