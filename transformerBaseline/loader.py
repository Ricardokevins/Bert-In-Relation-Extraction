import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from EasyTransformer import transformer
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

setup_seed(44)


def map_id_rel():
    id2rel={0: 'UNK', 1: '主演', 2: '歌手', 3: '简称', 4: '总部地点', 5: '导演', 6: '出生地', 7: '目', 8: '出生日期', 9: '占地面积', 10: '上映时间', 11: '出版社', 12: '作者', 13: '号', 14: '父亲', 15: '毕业院校', 16: '成立日期', 17: '改编自', 18: '主持人', 19: '所属专辑', 20: '连载网站', 21: '作词', 22: '作曲', 23: '创始人', 24: '丈夫', 25: '妻子', 26: '朝代', 27: '民族', 28: '国籍', 29: '身高', 30: '出品公司', 31: '母亲', 32: '编剧', 33: '首都', 34: '面积', 35: '祖籍', 36: '嘉宾', 37: '字', 38: '海拔', 39: '注册资本', 40: '制片人', 41: '董事长', 42: '所在城市', 43: '气候', 44: '人口数量', 45: '邮政编码', 46: '主角', 47: '官方语言', 48: '修业年限'}   
    rel2id={}
    for i in id2rel:
        rel2id[id2rel[i]]=i
    return rel2id,id2rel


def load_train():
    rel2id,id2rel=map_id_rel()
    max_length=128
    
    train_data = {}
    train_data['label'] = []
    train_data['mask'] = []
    train_data['text'] = []

    with open("../train.json", 'r', encoding='utf-8') as load_f:
        temp = load_f.readlines()
        lines = []
        temp=temp[:20000]
        for line in temp:
            dic = json.loads(line)
            if dic['rel'] not in rel2id:
                train_data['label'].append(0)
            else:
                train_data['label'].append(rel2id[dic['rel']])
            sent = dic['ent1'] + dic['ent2'] + dic['text']
            tokens = [i for i in sent]
            tokens=" ".join(tokens)
            lines.append(tokens)
        tokenizer = transformer.TransformerTokenizer(30000,256,lines)
        for sent in lines:
            indexed_tokens = tokenizer.encode(sent)        
            train_data['text'].append(indexed_tokens)

    return train_data

