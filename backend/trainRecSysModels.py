#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 19:07:07 2019

@author: latentlab
"""
import numpy as np
import pandas as pd

import sklearn
from sklearn.preprocessing import LabelEncoder

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

#%% load all the required data sets to train RecSys models
# load user-like pairs

male_userLikePair = pd.read_csv("processedData/male_userLikePair.csv")
female_userLikePair = pd.read_csv("processedData/female_userLikePair.csv")
all_userLikePair = pd.read_csv("processedData/all_userLikePair.csv")


#%%
# Collaborative Filtering
# use embedding to build a simple recommendation system
# Source:
# 1. Collaborative filtering, https://github.com/devforfu/pytorch_playground/blob/master/movielens.ipynb
# 2. https://github.com/yanneta/pytorch-tutorials/blob/master/collaborative-filtering-nn.ipynb

# here is a handy function modified from fast.ai
def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids. 
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)

def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["userID","like_id"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df

# training neural network based collaborative filtering
# neural network model
class collabFilterNet(nn.Module):
    def __init__(self, num_users, num_likes, embed_size, num_hidden):
        super(collabFilterNet, self).__init__()
        self.user_emb = nn.Embedding(num_users, embed_size)
        self.like_emb = nn.Embedding(num_likes,embed_size)
        self.fc1 = nn.Linear(embed_size*2, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 1)
        self.drop1 = nn.Dropout(0.1)
    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.like_emb(v)
        out = F.relu(torch.cat([U,V], dim=1))
        out = self.drop1(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
# training
def train_epocs(model,df_train, epochs, lr, wd, unsqueeze=False):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for i in range(epochs):
        users = torch.LongTensor(df_train.userID.values) # .cuda()
        items = torch.LongTensor(df_train.like_id.values) #.cuda()
        ratings = torch.FloatTensor(df_train.likes.values) #.cuda()
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

embed_size = 100
num_hidden = 10
#%%
# male recommender systems
# encoding data
maleUserLike_train = encode_data(male_userLikePair)

num_usersMale = len(maleUserLike_train.userID.unique())
num_likesMale = len(maleUserLike_train.like_id.unique())

model_maleUsers = collabFilterNet(num_usersMale, num_likesMale, embed_size, num_hidden)

train_epocs(model_maleUsers,df_train=maleUserLike_train, epochs=20, lr=0.001, wd=1e-6, unsqueeze=True)

torch.save(model_maleUsers.state_dict(), "trainedModel/model_maleUsers")



RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)
# female recommender systems
# encoding data
femaleUserLike_train = encode_data(female_userLikePair)

num_usersFemale = len(femaleUserLike_train.userID.unique())
num_likesFemale = len(femaleUserLike_train.like_id.unique())

model_femaleUsers = collabFilterNet(num_usersFemale, num_likesFemale, embed_size, num_hidden)

train_epocs(model_femaleUsers,df_train=femaleUserLike_train, epochs=20, lr=0.001, wd=1e-6, unsqueeze=True)

torch.save(model_femaleUsers.state_dict(), "trainedModel/model_femaleUsers")



RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)
# all users recommender systems
# encoding data
allUserLike_train = encode_data(all_userLikePair)

num_usersAll = len(allUserLike_train.userID.unique())
num_likesAll = len(allUserLike_train.like_id.unique())

model_allUsers = collabFilterNet(num_usersAll, num_likesAll, embed_size, num_hidden)

train_epocs(model_allUsers,df_train=allUserLike_train, epochs=20, lr=0.001, wd=1e-6, unsqueeze=True)

torch.save(model_allUsers.state_dict(), "trainedModel/model_allUsers")