import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import warnings
import torch
import time
import argparse
import json
import os
from transformers import BertTokenizer
from model import BERT_Classifier


#from transformers import BertPreTrainedModel
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
warnings.filterwarnings("ignore")
setup_seed(44)

from transformers import BertModel

from loader import map_id_rel

rel2id, id2rel = map_id_rel()

print(len(rel2id))
print(id2rel)

USE_CUDA = torch.cuda.is_available()


def test(net_path,text_list,ent1_list,ent2_list,result,show_result=False):
    max_length=128
    net=torch.load(net_path)
    net.eval()
    if USE_CUDA:
        net = net.cuda()
    rel_list = []
    correct=0
    total=0
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
            result = predicted.cpu().numpy().tolist()[0]
            if show_result:
                print("Source Text: ",text)
                print("Entity1: ",ent1," Entity2: ",ent2," Predict Relation: ",id2rel[result]," True Relation: ",label)
            if id2rel[result]==label:
                correct+=1
            total+=1
            #print('\n')
            rel_list.append(id2rel[result])
    print(correct," ",total," ",correct/total)
    return rel_list




from random import choice

def demo_output():
    text_list=[]
    ent1=[]
    ent2=[]
    result=[]
    total_num=3
    with open("train.json", 'r', encoding='utf-8') as load_f:
        lines=load_f.readlines()       
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
    test('0.9537394662921348.pth', text_list, ent1, ent2, result,True)

# 计算每一个类别的正确率
def caculate_acc():
    for i in range(len(rel2id)):
        temp_rel = id2rel[i]
        text_list=[]
        ent1=[]
        ent2=[]
        result=[]
        with open("dev.json", 'r', encoding='utf-8') as load_f:
            lines = load_f.readlines()
            for line in lines:
                line=choice(lines)
                dic = json.loads(line)
                if dic['rel']==temp_rel:
                    text_list.append(dic['text'])
                    ent1.append(dic['ent1'])
                    ent2.append(dic['ent2'])
                    result.append(dic['rel'])
                if len(text_list)==100:
                    break
        if len(text_list) == 0:
            print("No sample: ", temp_rel)
        else:
            test('0.9537394662921348.pth', text_list, ent1, ent2, result)

demo_output()
caculate_acc()