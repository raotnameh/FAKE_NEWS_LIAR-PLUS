from collections import OrderedDict 
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
import numpy as np
import torch


def words_token(sent,word_to_ix):
    list_ = np.zeros(shape=(len(sent),1)).astype('float64')
    for i,data in enumerate(sent):
        list_[i] = word_to_ix[data]
    return list_


# use one of the label_token for specific classification task

def label_token(label): # for 2 way classification
    mapping = {'true': 1, 'mostly-true': 1, 'half-true': 1, 'barely-true': 0, 'false': 0, 'pants-fire': 0}
    return [mapping[label]]

# def label_token(label): # for 6 way classification
#     mapping = {'true': 5, 'mostly-true': 4, 'half-true': 3, 'barely-true': 2, 'false': 1, 'pants-fire': 0}
#     return [mapping[label]]

def into_token(direct,word_to_ix):
    train_= []
    for i in direct:
        word_ = words_token(i[0],word_to_ix)
        label_ = label_token(i[1])
        train_.append([word_,label_])
    
    return np.array(train_)


def word_to_ix_(data):
    word_to_ix = OrderedDict()
    for sent in data :
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix

def label_to_ix_(data):
    label_to_ix = OrderedDict()
    for sent in data:
        for label in sent:
            if label not in label_to_ix:
                label_to_ix[label] = len(label_to_ix)
    return label_to_ix

def make_target(label,label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

def sent_words(data,label = False):
    lis = []
    if label == False:
        for i in range(len(data)):
            lis.append(data['statement'][i].split())
    else:
        for i in range(len(data)):
            lis.append(data['label'][i].split())
    return np.array(lis)

def one_hot_(sent):
    words = torch.tensor([word_to_ix[w] for w in sent ], dtype=torch.long)
    print(words)
    one_hot_encoding = one_hot(words)
    return one_hot_encoding