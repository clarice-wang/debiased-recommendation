#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:15:58 2019

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

import sklearn as sk
from sklearn.preprocessing import LabelEncoder,normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import PCA



#%% 
# Load input user-like pair to test all the models
testUserLikePair = pd.read_csv("input_output/testUserLikePair.csv")



#%%
# encoding like as rating 1 and no-like as no rating
likes=np.ones((testUserLikePair.shape[0]))
likes = pd.DataFrame(likes,columns=['likes'])
testUserLikePair = pd.concat([testUserLikePair,likes],axis=1) 

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

male_userID = np.int64(np.loadtxt('processedData/male_userID.txt'))
female_userID = np.int64(np.loadtxt('processedData/female_userID.txt'))
all_userID = np.int64(np.loadtxt('processedData/all_userID.txt'))

userDemog = pd.read_csv("processedData/userDemog.csv")
userConcentrationsName = pd.read_csv("processedData/userConcentrationsName.csv")

#testUser = (all_userLikePair.loc[all_userLikePair['like_id'].isin(female_userLikePair['like_id'])]).reset_index(drop=True)
#testUser = (testUser.loc[testUser['like_id'].isin(male_userLikePair['like_id'])]).reset_index(drop=True)
#
#dictionary_commonLikesID = testUser.like_id.unique()
#np.savetxt('dictionary_commonLikesID.txt',dictionary_commonLikesID)
#    
#    
#msk = np.random.rand(len(testUser)) < 0.8
#trainPair = testUser[msk].copy()
#valPair = testUser[~msk].copy()
#
#testUserLikePair = trainPair[5000:7000].reset_index(drop=True)
#testUserLikePair.to_csv('testUserLikePair.csv',index=False)

#%%
# encoding
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
    for col_name in ["like_id"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df

def encode_user(df, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["userID"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df

def nearest_recommend(testEmbed,userEmbed,usersID,userConcentrationsName):
    # assign k similar users for each test users
    user_distance = pairwise_distances(testEmbed,userEmbed,metric='cosine')
    
    k = 10
    user_recCon = []
    
    for i in range(len(user_distance)):
        user_similarity = np.argsort(user_distance[i])[:k]
        temp_concen1 = []
        for j in range(k):
            similar_user1 = usersID[user_similarity[j]]
            tmp1 = (userConcentrationsName.loc[userConcentrationsName['userID']==similar_user1]).reset_index(drop=True)
            temp_concen1.append(tmp1['name'].values[0])
                
        user_recCon.append(temp_concen1)
    return user_recCon

def logisticRegression_recommend(testEmbed,userEmbed,usersID,userConcentrationsName):
    # standardize the data
    scaler = StandardScaler().fit(userEmbed)
    userEmbed = scaler.transform(userEmbed)
    testEmbed  = scaler.transform(testEmbed)
    
    concentrationList = (pd.concat([userConcentrationsName['concentration_id'],userConcentrationsName['name']],axis=1)).reset_index(drop=True) 
    # creating data labels
    labels = np.int64(np.zeros(len(userEmbed)))
    for i in range(len(userEmbed)):
        u = usersID[i]
        tmp = (userConcentrationsName.loc[userConcentrationsName['userID']==u]).reset_index(drop=True)
        labels[i] = tmp['concentration_id'].values[0]
        
    # classification
    clf = LogisticRegression(random_state=0,max_iter = 200, C=0.00001,
                             solver = 'lbfgs', multi_class = 'multinomial').fit(userEmbed, labels)
    
    # dropping bias term
    clf.intercept_ = np.zeros((len(clf.intercept_)))
    predictedLabel = clf.predict(testEmbed)
    
    user_recCon = []
    for i in range(len(testEmbed)):
        user_recCon.append((concentrationList[concentrationList['concentration_id']==predictedLabel[i]].values)[0][1])
    
    return user_recCon

#%% Collaborative Filtering
# use embedding to build a simple recommendation system
# Source:
# 1. Collaborative filtering, https://github.com/devforfu/pytorch_playground/blob/master/movielens.ipynb
# 2. https://github.com/yanneta/pytorch-tutorials/blob/master/collaborative-filtering-nn.ipynb
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
    model.like_emb.weight.requires_grad = False
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
        #print(loss.item())

embed_size = 100
num_hidden = 10

#%%
# Male recommender systems
# encoding data
print("Evaluating male recommender system")
maleUserLike_train = encode_data(male_userLikePair)
maleUserLike_train = encode_user(maleUserLike_train)

testMale = encode_data(testUserLikePair,male_userLikePair)
testMale = encode_user(testMale)

num_usersMale = len(maleUserLike_train.userID.unique())
num_likesMale = len(maleUserLike_train.like_id.unique())

model_maleUsers = collabFilterNet(num_usersMale, num_likesMale, embed_size, num_hidden)


model_maleUsers.load_state_dict(torch.load("trainedModel/model_maleUsers"))
maleUsersEmbed = (model_maleUsers.user_emb.weight).detach().numpy()

model_maleUsers.user_emb = nn.Embedding(len(testMale.userID.unique()), embed_size)

train_epocs(model_maleUsers,df_train=testMale, epochs=25, lr=0.001, wd=1e-6, unsqueeze=True)
testmaleEmbed = (model_maleUsers.user_emb.weight).detach().numpy()

# normalizing embedding
maleUsersEmbed = maleUsersEmbed / np.linalg.norm(maleUsersEmbed,axis=1,keepdims=1)
testmaleEmbed = testmaleEmbed / np.linalg.norm(testmaleEmbed,axis=1,keepdims=1)

# nearest neighbored based recommendations
maleRecSys_nearest = nearest_recommend(testmaleEmbed,maleUsersEmbed,male_userID,userConcentrationsName) 
maleRecSys_nearest = pd.DataFrame(np.asarray(maleRecSys_nearest))
maleRecSys_nearest.to_csv('input_output/maleRecSys_nearest.csv')

# logistic regression based recommendations (drop bias term)
maleRecSys_LR = logisticRegression_recommend(testmaleEmbed,maleUsersEmbed,male_userID,userConcentrationsName)
maleRecSys_LR = pd.DataFrame(np.asarray(maleRecSys_LR))
maleRecSys_LR.to_csv('input_output/maleRecSys_LR.csv')


#%%
# Female recommender systems
# encoding data
print("Evaluating female recommender system")
femaleUserLike_train = encode_data(female_userLikePair)
femaleUserLike_train = encode_user(femaleUserLike_train)

testFemale = encode_data(testUserLikePair,female_userLikePair)
testFemale = encode_user(testFemale)

num_usersFemale = len(femaleUserLike_train.userID.unique())
num_likesFemale = len(femaleUserLike_train.like_id.unique())

model_femaleUsers = collabFilterNet(num_usersFemale, num_likesFemale, embed_size, num_hidden)


model_femaleUsers.load_state_dict(torch.load("trainedModel/model_femaleUsers"))
femaleUsersEmbed = (model_femaleUsers.user_emb.weight).detach().numpy()

model_femaleUsers.user_emb = nn.Embedding(len(testFemale.userID.unique()), embed_size)

train_epocs(model_femaleUsers,df_train=testFemale, epochs=25, lr=0.001, wd=1e-6, unsqueeze=True)
testfemaleEmbed = (model_femaleUsers.user_emb.weight).detach().numpy()

# normalizing embedding
femaleUsersEmbed = femaleUsersEmbed / np.linalg.norm(femaleUsersEmbed,axis=1,keepdims=1)
testfemaleEmbed = testfemaleEmbed / np.linalg.norm(testfemaleEmbed,axis=1,keepdims=1)

# nearest neighbored based recommendations
femaleRecSys_nearest = nearest_recommend(testfemaleEmbed,femaleUsersEmbed,female_userID,userConcentrationsName) 
femaleRecSys_nearest = pd.DataFrame(np.asarray(femaleRecSys_nearest))
femaleRecSys_nearest.to_csv('input_output/femaleRecSys_nearest.csv')

# logistic regression based recommendations (drop bias term)
femaleRecSys_LR = logisticRegression_recommend(testfemaleEmbed,femaleUsersEmbed,female_userID,userConcentrationsName)
femaleRecSys_LR = pd.DataFrame(np.asarray(femaleRecSys_LR))
femaleRecSys_LR.to_csv('input_output/femaleRecSys_LR.csv')

#%%
# Typical recommender systems
# encoding data
print("Evaluating typical recommender system (male+female users without debiasing)")
allUserLike_train = encode_data(all_userLikePair)
allUserLike_train = encode_user(allUserLike_train)

testAll = encode_data(testUserLikePair,all_userLikePair)
testAll = encode_user(testAll)

num_usersAll = len(allUserLike_train.userID.unique())
num_likesAll = len(allUserLike_train.like_id.unique())

model_allUsers = collabFilterNet(num_usersAll, num_likesAll, embed_size, num_hidden)


model_allUsers.load_state_dict(torch.load("trainedModel/model_allUsers"))
allUsersEmbed = (model_allUsers.user_emb.weight).detach().numpy()

model_allUsers.user_emb = nn.Embedding(len(testAll.userID.unique()), embed_size)

train_epocs(model_allUsers,df_train=testAll, epochs=25, lr=0.001, wd=1e-6, unsqueeze=True)
testallEmbed = (model_allUsers.user_emb.weight).detach().numpy()

# normalizing embedding
allUsersEmbed = allUsersEmbed / np.linalg.norm(allUsersEmbed,axis=1,keepdims=1)
testallEmbed = testallEmbed / np.linalg.norm(testallEmbed,axis=1,keepdims=1)

# nearest neighbored based recommendations
typicalRecSys_nearest = nearest_recommend(testallEmbed,allUsersEmbed,all_userID,userConcentrationsName) 
typicalRecSys_nearest = pd.DataFrame(np.asarray(typicalRecSys_nearest))
typicalRecSys_nearest.to_csv('input_output/typicalRecSys_nearest.csv')

# logistic regression based recommendations (drop bias term)
typicalRecSys_LR = logisticRegression_recommend(testallEmbed,allUsersEmbed,all_userID,userConcentrationsName)
typicalRecSys_LR = pd.DataFrame(np.asarray(typicalRecSys_LR))
typicalRecSys_LR.to_csv('input_output/typicalRecSys_LR.csv')


#%%
# Gender-neutral recommender systems
# encoding data
print("Evaluating gender-neutral recommender system")

# bias direction
genderEmbed = np.zeros((2,allUsersEmbed.shape[1]))
for i in range(len(allUsersEmbed)):
    u = all_userID[i]
    if userDemog['gender'][u] == 0:
        genderEmbed[0] +=  allUsersEmbed[i] 
    else:
        genderEmbed[1] +=  allUsersEmbed[i] 

genderEmbed = genderEmbed / np.linalg.norm(genderEmbed,axis=1,keepdims=1)
vBias= genderEmbed[1]-genderEmbed[0]
vBias = vBias.reshape(1,-1)
vBias = vBias / np.linalg.norm(vBias,axis=1,keepdims=1)

# linear projection: u - <u,v_b>v_b
debiased_userEmbed = np.zeros((len(allUsersEmbed),allUsersEmbed.shape[1]))
for i in range(len(allUsersEmbed)):
    debiased_userEmbed[i] = allUsersEmbed[i] - (np.inner(allUsersEmbed[i].reshape(1,-1),vBias)[0][0])*vBias
    
debiased_testEmbed = np.zeros((len(testallEmbed),testallEmbed.shape[1]))
for i in range(len(testallEmbed)):
    debiased_testEmbed[i] = testallEmbed[i] - (np.inner(testallEmbed[i].reshape(1,-1),vBias)[0][0])*vBias

# nearest neighbored based recommendations
debiasRecSys_nearest = nearest_recommend(debiased_testEmbed,debiased_userEmbed,all_userID,userConcentrationsName) 
debiasRecSys_nearest = pd.DataFrame(np.asarray(debiasRecSys_nearest))
debiasRecSys_nearest.to_csv('input_output/debiasRecSys_nearest.csv')

# logistic regression based recommendations (drop bias term)
debiasRecSys_LR = logisticRegression_recommend(debiased_testEmbed,debiased_userEmbed,all_userID,userConcentrationsName)
debiasRecSys_LR = pd.DataFrame(np.asarray(debiasRecSys_LR))
debiasRecSys_LR.to_csv('input_output/debiasRecSys_LR.csv')