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
from transformers import AdamW
import warnings
import torch
import time
import argparse
import json
import os
from EasyTransformer import transformer
from model import Model

from loader import load_train
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



USE_CUDA = torch.cuda.is_available()
#USE_CUDA=False

data=load_train()
train_text=data['text']
train_label=data['label']



train_text=torch.tensor(train_text)

train_label=torch.tensor(train_label)

print("--train data--")
print(train_text.shape)

print(train_label.shape)





# exit()
#USE_CUDA=False

if USE_CUDA:
    print("using GPU")

train_dataset = torch.utils.data.TensorDataset(train_text,train_label)


net= Model()




def train(net,dataset,num_epochs, learning_rate,  batch_size):
    net.train()
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate,weight_decay=0)
    optimizer = AdamW(net.parameters(), lr=learning_rate)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    pre = 0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        correct = 0
        total=0
        iter = 0
        for text,y in train_iter:
            iter += 1
            optimizer.zero_grad()
            if text.size(0)!=batch_size:
                break
            text=text.reshape(batch_size,-1)
            
            if USE_CUDA:
                text=text.cuda()
                y = y.cuda()
            #print(text.shape)
            state= net(text)
            #print(y)
            #print(loss.shape)
            #print("predicted",predicted)
            #print("answer", y)
            loss = criterion(state, y)
            loss.backward()
            optimizer.step()
            #print(outputs[1].shape)
            #print(outputs[1])
            correct += state.argmax(dim=-1).eq(y).sum().item()
            total += text.size(0)

        loss = loss.detach().cpu()
        print(correct)
        print(total)
        print(correct/total)
        #print("epoch ", str(epoch)," loss: ", loss.mean().numpy().tolist(),"right", correct.cpu().numpy().tolist(), "total", total, "Acc:", correct.cpu().numpy().tolist()/total)
    return


#model=nn.DataParallel(model,device_ids=[0,1])
if USE_CUDA:
    net=net.cuda()

#eval(model,dev_dataset,8)

train(net,train_dataset,10,2e-3,32)
#eval(model,dev_dataset,8)

