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
import time
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

import pickle

#%% 

def initialize_test():
    # Load input user-like pair to test all the models
    testUserLikePair = pd.read_csv("input_output/testUserLikePair.csv")
    # encoding like as rating 1 and no-like as no rating
    likes=np.ones((testUserLikePair.shape[0]))
    likes = pd.DataFrame(likes,columns=['likes'])
    testUserLikePair = pd.concat([testUserLikePair,likes],axis=1) 
    return testUserLikePair

#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

def initialize():
    RANDOM_STATE = 1
    set_random_seed(RANDOM_STATE)

    #%% load all the required data sets to train RecSys models
    # load user-like pairs
    print("Loading data ...")
    male_userLikePair = pd.read_csv("processedData/male_userLikePair.csv")
    female_userLikePair = pd.read_csv("processedData/female_userLikePair.csv")
    all_userLikePair = pd.read_csv("processedData/all_userLikePair.csv")

    male_userID = np.int64(np.loadtxt('processedData/male_userID.txt'))
    female_userID = np.int64(np.loadtxt('processedData/female_userID.txt'))
    all_userID = np.int64(np.loadtxt('processedData/all_userID.txt'))

    userDemog = pd.read_csv("processedData/userDemog.csv")
    userConcentrationsName = pd.read_csv("processedData/userConcentrationsName.csv")
    print("Loading data ... done")

    return (male_userLikePair, female_userLikePair, all_userLikePair, male_userID, female_userID, all_userID, userDemog, userConcentrationsName)

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

def logisticRegression_recommend(testEmbed,userEmbed,usersID,userConcentrationsName,clf):
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
    
    uniqueLabels=np.unique(labels)
# =============================================================================
#     # classification
#     clf = LogisticRegression(random_state=0,max_iter = 500, C=0.0001,
#                              solver = 'sag', multi_class = 'multinomial').fit(userEmbed, labels)
#     
#     # dropping bias term
#     clf.intercept_ = np.zeros((len(clf.intercept_)))
#     # save the model to disk
#     filename = 'trainedModel/logisticRegressionGenderNeutral.sav'
#     pickle.dump(clf, open(filename, 'wb'))
# =============================================================================

    predictProb = clf.predict_proba(testEmbed)
    k=10
    
    user_recCon = []
    for i in range(len(predictProb)):
        topIndx = np.argsort(1-predictProb[i])[:k]
        temp_concen1 = []
        for j in range(k):
            temp_concen1.append((concentrationList[concentrationList['concentration_id']==uniqueLabels[topIndx[j]]].values)[0][1])
        user_recCon.append(temp_concen1)
    
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
print("Evaluating male recommender system")

def initialize_model_embed(userLikePair, model_file):
    # Male recommender systems
    # encoding data
    UserLike_train = encode_data(userLikePair)
    UserLike_train = encode_user(UserLike_train)
    num_users = len(UserLike_train.userID.unique())
    num_likes = len(UserLike_train.like_id.unique())

    model_Users = collabFilterNet(num_users, num_likes, embed_size, num_hidden)
    model_Users.load_state_dict(torch.load(model_file))
    UsersEmbed = (model_Users.user_emb.weight).detach().numpy()
    # normalizing embedding
    UsersEmbed = UsersEmbed / np.linalg.norm(UsersEmbed,axis=1,keepdims=1)
    
    return (model_Users, UsersEmbed)

def eval_recommender(testUserLikePair, userLikePair, model_Users, UsersEmbed, userID, userConcentrationsName, model_file, output_file):
    test = encode_data(testUserLikePair, userLikePair)
    test = encode_user(test)

    start = time.time()
    model_Users.user_emb = nn.Embedding(len(test.userID.unique()), embed_size)
    print("embedding time: ", time.time() - start)

    start = time.time()
    train_epocs(model_Users,df_train=test, epochs=25, lr=0.001, wd=1e-6, unsqueeze=True)
    print("train_epocs time: ", time.time() - start)

    start = time.time()
    testEmbed = (model_Users.user_emb.weight).detach().numpy()
    # normalizing embedding
    testEmbed = testEmbed / np.linalg.norm(testEmbed,axis=1,keepdims=1)
    print("normalizing time: ", time.time() - start)

# nearest neighbored based recommendations
# maleRecSys_nearest = nearest_recommend(testmaleEmbed,maleUsersEmbed,male_userID,userConcentrationsName) 
# nmaleRecSys_nearest = pd.DataFrame(np.asarray(maleRecSys_nearest))
# maleRecSys_nearest.to_csv('input_output/maleRecSys_nearest.csv')

# logistic regression based recommendations (drop bias term)
    start = time.time()
    Clf = pickle.load(open(model_file, 'rb'))
    print("model loading time: ", time.time() - start)

    start = time.time()
    RecSys_LR = logisticRegression_recommend(testEmbed,UsersEmbed,userID,userConcentrationsName,Clf)
    RecSys_LR = pd.DataFrame(np.asarray(RecSys_LR))
    RecSys_LR.to_csv(output_file)
    print("regression time: ", time.time() - start)

print("initializing ...")
(male_userLikePair, female_userLikePair, all_userLikePair, male_userID, female_userID, all_userID, userDemog, userConcentrationsName) = initialize()
(model_maleUsers, maleUsersEmbed) = initialize_model_embed(male_userLikePair, "trainedModel/model_maleUsers")
(model_femaleUsers, femaleUsersEmbed) = initialize_model_embed(female_userLikePair, "trainedModel/model_femaleUsers")
(model_allUsers, allUsersEmbed) = initialize_model_embed(all_userLikePair, "trainedModel/model_allUsers")

print("starting test ...")

time1 = time.time()

testUserLikePair = initialize_test()
eval_recommender(testUserLikePair, male_userLikePair, model_maleUsers, maleUsersEmbed, male_userID, userConcentrationsName, "trainedModel/logisticRegressionMale.sav", "input_output/maleRecSys_LR.csv")
eval_recommender(testUserLikePair, female_userLikePair, model_femaleUsers, femaleUsersEmbed, female_userID, userConcentrationsName, "trainedModel/logisticRegressionFemale.sav", "input_output/femaleRecSys_LR.csv")
eval_recommender(testUserLikePair, all_userLikePair, model_allUsers, allUsersEmbed, all_userID, userConcentrationsName, "trainedModel/logisticRegressionTypical.sav", "input_output/typicalRecSys_LR.csv")

time2 = time.time()

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

testallEmbed = (model_allUsers.user_emb.weight).detach().numpy()
testallEmbed = testallEmbed / np.linalg.norm(testallEmbed,axis=1,keepdims=1)

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
# debiasRecSys_nearest = nearest_recommend(debiased_testEmbed,debiased_userEmbed,all_userID,userConcentrationsName) 
# debiasRecSys_nearest = pd.DataFrame(np.asarray(debiasRecSys_nearest))
# debiasRecSys_nearest.to_csv('input_output/debiasRecSys_nearest.csv')

# logistic regression based recommendations (drop bias term)
filename = 'trainedModel/logisticRegressionGenderNeutral.sav'
clf_neutral = pickle.load(open(filename, 'rb'))

debiasRecSys_LR = logisticRegression_recommend(debiased_testEmbed,debiased_userEmbed,all_userID,userConcentrationsName,clf_neutral)
debiasRecSys_LR = pd.DataFrame(np.asarray(debiasRecSys_LR))
debiasRecSys_LR.to_csv('input_output/debiasRecSys_LR.csv')


time3 = time.time()

print("elapsed time for male/female/all LR models: ", time2-time1)
print("elapsed time for debiased LR models: ", time3-time2)

