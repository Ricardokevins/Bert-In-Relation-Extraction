import json
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

setup_seed(44)
def prepare_data():
    with open("train_data.json", 'r', encoding='utf-8') as load_f:
        info=[]
        import random
        for line in load_f.readlines():
            dic = json.loads(line)
            for j in dic['spo_list']:
                single_data={}
                single_data['rel']=j["predicate"]
                single_data['ent1']=j["object"]
                single_data['ent2'] = j["subject"]
                single_data['text']=dic['text']
                info.append(single_data)
        sub_train = []
        for i in range(20000):
            sub_train.append(random.choice(info))
    with open("train.json", "w",encoding='utf-8') as dump_f:
        for i in sub_train:
            a = json.dumps(i, ensure_ascii=False)
            dump_f.write(a)
            dump_f.write("\n")
    
    with open("dev_data.json", 'r', encoding='utf-8') as load_f:
        info=[]
        import random
        for line in load_f.readlines():
            dic = json.loads(line)
            for j in dic['spo_list']:
                single_data={}
                single_data['rel']=j["predicate"]
                single_data['ent1']=j["object"]
                single_data['ent2'] = j["subject"]
                single_data['text']=dic['text']
                info.append(single_data)
            
        sub_train = []
        for i in range(2000):
            sub_train.append(random.choice(info))
    with open("dev.json", "w",encoding='utf-8') as dump_f:
        for i in sub_train:
            a = json.dumps(i, ensure_ascii=False)
            dump_f.write(a)
            dump_f.write("\n")



# def map_id_rel():
#     rel = ["UNK"]
#     with open("train.json", 'r', encoding='utf-8') as load_f:
#         for line in load_f.readlines():
#             dic = json.loads(line)
#             if dic['rel'] not in rel:
#                 rel.append(dic['rel'])
#     id2rel={}
#     rel2id={}
#     for i in range(len(rel)):
#         id2rel[i]=rel[i]
#         rel2id[rel[i]]=i
#     return rel2id,id2rel

def map_id_rel():
    id2rel={0: 'UNK', 1: '主演', 2: '歌手', 3: '简称', 4: '总部地点', 5: '导演', 6: '出生地', 7: '目', 8: '出生日期', 9: '占地面积', 10: '上映时间', 11: '出版社', 12: '作者', 13: '号', 14: '父亲', 15: '毕业院校', 16: '成立日期', 17: '改编自', 18: '主持人', 19: '所属专辑', 20: '连载网站', 21: '作词', 22: '作曲', 23: '创始人', 24: '丈夫', 25: '妻子', 26: '朝代', 27: '民族', 28: '国籍', 29: '身高', 30: '出品公司', 31: '母亲', 32: '编剧', 33: '首都', 34: '面积', 35: '祖籍', 36: '嘉宾', 37: '字', 38: '海拔', 39: '注册资本', 40: '制片人', 41: '董事长', 42: '所在城市', 43: '气候', 44: '人口数量', 45: '邮政编码', 46: '主角', 47: '官方语言', 48: '修业年限'}   
    rel2id={}
    for i in id2rel:
        rel2id[id2rel[i]]=i
    return rel2id,id2rel

def load_train():
    rel2id,id2rel=map_id_rel()
    max_length=128
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_data = {}
    train_data['label'] = []
    train_data['mask'] = []
    train_data['text'] = []

    with open("train.json", 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            dic = json.loads(line)
            if dic['rel'] not in rel2id:
                train_data['label'].append(0)
            else:
                train_data['label'].append(rel2id[dic['rel']])
            sent=dic['ent1']+dic['ent2']+dic['text']
            indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
            avai_len = len(indexed_tokens)
            while len(indexed_tokens) <  max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

            # Attention mask
            att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            train_data['text'].append(indexed_tokens)
            train_data['mask'].append(att_mask)
    return train_data

def load_dev():
    rel2id,id2rel=map_id_rel()
    max_length=128
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_data = {}
    train_data['label'] = []
    train_data['mask'] = []
    train_data['text'] = []

    with open("dev.json", 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            dic = json.loads(line)
            if dic['rel'] not in rel2id:
                train_data['label'].append(0)
            else:
                train_data['label'].append(rel2id[dic['rel']])

            sent=dic['ent1']+dic['ent2']+dic['text']
            indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
            avai_len = len(indexed_tokens)
            while len(indexed_tokens) <  max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

            # Attention mask
            att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            train_data['text'].append(indexed_tokens)
            train_data['mask'].append(att_mask)
    return train_data