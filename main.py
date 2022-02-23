#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       : main function to carry out training and testing
@Author             : Kevinpro
@version            : 1.0
'''
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
from transformers import AdamW
import warnings
import torch
import time
import argparse
import json
import os
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

#from transformers import BertPreTrainedModel


from transformers import BertModel

from loader import load_train
from loader import load_dev

from loader import map_id_rel
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

setup_seed(44)

rel2id, id2rel = map_id_rel()

print(len(rel2id))
print(id2rel)

def get_model():
    labels_num=len(rel2id)
    from model import BERT_Classifier
    model = BERT_Classifier(labels_num)
    return model

model=get_model()
# torch.save(model, './bert-base-chinese/test'+'.pth')
# exit()
# exit()
USE_CUDA = torch.cuda.is_available()
#USE_CUDA=False

data=load_train()
train_text=data['text']
train_mask=data['mask']
train_label=data['label']

train_text = [ t.numpy() for t in train_text]
train_mask = [ t.numpy() for t in train_mask]

train_text=torch.tensor(train_text)
train_mask=torch.tensor(train_mask)
train_label=torch.tensor(train_label)

print("--train data--")
print(train_text.shape)
print(train_mask.shape)
print(train_label.shape)

data=load_dev()
dev_text=data['text']
dev_mask=data['mask']
dev_label=data['label']

dev_text = [ t.numpy() for t in dev_text]
dev_mask = [ t.numpy() for t in dev_mask]

dev_text=torch.tensor(dev_text)
dev_mask=torch.tensor(dev_mask)
dev_label=torch.tensor(dev_label)

print("--train data--")
print(train_text.shape)
print(train_mask.shape)
print(train_label.shape)

print("--eval data--")
print(dev_text.shape)
print(dev_mask.shape)
print(dev_label.shape)

# exit()
#USE_CUDA=False

if USE_CUDA:
    print("using GPU")

train_dataset = torch.utils.data.TensorDataset(train_text,train_mask,train_label)
dev_dataset = torch.utils.data.TensorDataset(dev_text,dev_mask,dev_label)




def eval(net,dataset, batch_size):
    net.eval()
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    with torch.no_grad():
        correct = 0
        total=0
        iter = 0
        for text,mask, y in train_iter:
            iter += 1
            if text.size(0)!=batch_size:
                break
            text=text.reshape(batch_size,-1)
            mask = mask.reshape(batch_size, -1)
            
            if USE_CUDA:
                text=text.cuda()
                mask=mask.cuda()
                y=y.cuda()

            outputs= net(text, mask,y)
            #print(y)
            loss, logits = outputs[0],outputs[1]
            _, predicted = torch.max(logits.data, 1)
            total += text.size(0)
            correct += predicted.data.eq(y.data).cpu().sum()
            s = ("Acc:%.3f" %((1.0*correct.numpy())/total))
        acc= (1.0*correct.numpy())/total
        print("Eval Result: right", correct.cpu().numpy().tolist(), "total", total, "Acc:", acc)
        return acc


def train(net,dataset,num_epochs, learning_rate,  batch_size):
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,weight_decay=0)
    #optimizer = AdamW(net.parameters(), lr=learning_rate)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    pre=0
    for epoch in range(num_epochs):
        correct = 0
        total=0
        iter = 0
        for text,mask, y in train_iter:
            iter += 1
            optimizer.zero_grad()
            #print(type(y))
            #print(y)
            if text.size(0)!=batch_size:
                break
            text=text.reshape(batch_size,-1)
            mask = mask.reshape(batch_size, -1)
            if USE_CUDA:
                text=text.cuda()
                mask=mask.cuda()
                y = y.cuda()
            #print(text.shape)
            loss, logits= net(text, mask,y)
            #print(y)
            #print(loss.shape)
            #print("predicted",predicted)
            #print("answer", y)
            loss.backward()
            optimizer.step()
            #print(outputs[1].shape)
            #print(output)
            #print(outputs[1])
            _, predicted = torch.max(logits.data, 1)
            total += text.size(0)
            correct += predicted.data.eq(y.data).cpu().sum()
        loss = loss.detach().cpu()
        print("epoch ", str(epoch)," loss: ", loss.mean().numpy().tolist(),"right", correct.cpu().numpy().tolist(), "total", total, "Acc:", correct.cpu().numpy().tolist()/total)
        acc = eval(model, dev_dataset, 32)
        if acc > pre:
            pre = acc
            torch.save(model, str(acc)+'.pth')
    return



#model=nn.DataParallel(model,device_ids=[0,1])
if USE_CUDA:
    model=model.cuda()

#eval(model,dev_dataset,8)

train(model,train_dataset,1,0.002,4)
#eval(model,dev_dataset,8)

