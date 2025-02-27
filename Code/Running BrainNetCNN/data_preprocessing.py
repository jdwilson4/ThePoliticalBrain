#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 22:14:17 2021

@author: Seo Eun Yang (se.yang@northeastern.edu)
"""
import csv, os, sys, pickle
import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn import preprocessing  
from sklearn.model_selection import train_test_split

#path='ThePoliticalBrain-main/Code/Running BrainNetCNN/'
#os.chdir(path)

#random seed
np.random.seed(seed=333) 
 
def create_new_ts(mat): 
    ### create correlation matrix from time series observation 
    ### output (1) corr: correlation matrix
    corr=np.zeros((mat.shape[0],mat.shape[2]-1,mat.shape[2]-1))
    exc252=[x for x in range(0,mat.shape[2]) if x != 252] #exclude 252th node (Nan)
    for j in range(len(mat)):
        df = pd.DataFrame(mat[j])
        cor_mat = df.corr(method='pearson')
        corr[j,:,:] = cor_mat.loc[exc252,exc252]
    return corr

def draw_data():
    ### extract information from .csv and .mat format for our own purpose
    ### output
    ### (1) conservative: ideology score
    ### (2) matched: matched survey data
    ### (3) cor_mats_all_tasks: correlation matrices of all nine tasks
    
    #load survey data which contains democraphic information such as age, ideology, gender 
    wb_info=pd.read_csv('wb_info_250.csv', sep=",")         
    info_id=wb_info['participant'].tolist() #subject ID in survey data
    
    #load matrix data
    mat = scio.loadmat('tc.mat')  #load all dataset in matlab format
    N = mat.get('tc').shape[0]       #Participants = 203 
    n = mat.get('tc')[0][0].shape[1] #Node = 269
    subj=[]                      #subject ID in matrix data
    for i in range(0,len(mat.get('subjs')[0])):
        subj.append(mat.get('subjs')[0][i][0])
    task=[]                      #task name
    for i in range(0,len(mat.get('tasks')[0])):
        task.append(mat.get('tasks')[0][i][0])
        
    #select unique IDs which exist in both survey data and matrix data
    match_loc_subj_short=[];match_loc_wb_short=[]
    for i in range(0,len(subj)):
        locs=[j for j,x in enumerate(info_id) if x == subj[i]]
        if len(locs) != 0: 
            match_loc_wb_short.append(locs[0])
            match_loc_subj_short.append(i)
    
    #include survey data if ID exists in both survey data and matrix data   
    matched = wb_info.loc[match_loc_wb_short,:]
    matched.index = range(0,len(matched))
    new_subj=np.array(subj)[match_loc_subj_short].tolist()  
    
    #conservative variable of right-handed people in survey data
    conservative=np.array(matched['conservative_you']).tolist()

    #extract matrix data from matlab file
    Affect=np.zeros((N,len(mat.get('tc')[0][0]),n))
    Empathy=np.zeros((N,len(mat.get('tc')[0][1]),n))
    Encoding=np.zeros((N,len(mat.get('tc')[0][2]),n))
    GoNogo=np.zeros((N,len(mat.get('tc')[0][3]),n))
    Resting=np.zeros((N,len(mat.get('tc')[0][4]),n))
    Retrieval=np.zeros((N,len(mat.get('tc')[0][5]),n))
    Reward=np.zeros((N,len(mat.get('tc')[0][6]),n))
    ToM=np.zeros((N,len(mat.get('tc')[0][7]),n))
    WorkingMem=np.zeros((N,len(mat.get('tc')[0][8]),n))
    for i in range(0,N):
        Affect[i,:,:]=mat.get('tc')[i][0]
        Empathy[i,:,:]=mat.get('tc')[i][1]
        Encoding[i,:,:]=mat.get('tc')[i][2] 
        GoNogo[i,:,:]=mat.get('tc')[i][3]
        Resting[i,:,:]=mat.get('tc')[i][4]
        Retrieval[i,:,:]=mat.get('tc')[i][5]
        Reward[i,:,:]=mat.get('tc')[i][6]
        ToM[i,:,:]=mat.get('tc')[i][7]
        WorkingMem[i,:,:]=mat.get('tc')[i][8]
    
    #create correlation matrix from time series dataset
    Affect_cor = create_new_ts(Affect)
    Empathy_cor = create_new_ts(Empathy)
    Encoding_cor = create_new_ts(Encoding)
    GoNogo_cor = create_new_ts(GoNogo)
    Resting_cor = create_new_ts(Resting)
    Retrieval_cor = create_new_ts(Retrieval)
    Reward_cor = create_new_ts(Reward)
    ToM_cor = create_new_ts(ToM)
    WorkingMem_cor = create_new_ts(WorkingMem)

    #contain people who have information in both survey data and matrix data
    rAffect     = Affect_cor[match_loc_subj_short,:,:]
    rEmpathy    = Empathy_cor[match_loc_subj_short,:,:]
    rEncoding   = Encoding_cor[match_loc_subj_short,:,:]
    rGoNogo     = GoNogo_cor[match_loc_subj_short,:,:]
    rResting    = Resting_cor[match_loc_subj_short,:,:]
    rRetrieval  = Retrieval_cor[match_loc_subj_short,:,:]
    rReward     = Reward_cor[match_loc_subj_short,:,:]
    rToM        = ToM_cor[match_loc_subj_short,:,:]
    rWorkingMem = WorkingMem_cor[match_loc_subj_short,:,:] 
    
    cor_mats_all_tasks = {'Affect':rAffect,'Empathy':rEmpathy,'Encoding':rEncoding,'GoNogo':rGoNogo,'Resting':rResting,'Retrieval':rRetrieval,'Reward':rReward,'ToM':rToM,'WorkingMem':rWorkingMem}
    matched.to_csv('wb_info_250_matched.csv')
    return conservative, matched, cor_mats_all_tasks
      
def make_test_train_valid_for4CV(y_test1,y_test2,y_test3,y_test4,test_ind1,test_ind2,test_ind3,test_ind4,conservative):
    ### split test, train, and valid set 
    ### output (1) com: a dictionary containing train, test, and valid dataset and their indices.
    y_combine = np.concatenate([y_test1,y_test2,y_test3])
    combine_ind = np.concatenate([test_ind1,test_ind2,test_ind3]) 
    test_ind = test_ind4
    y_train, y_valid, train_ind, valid_ind = train_test_split(y_combine, combine_ind, stratify = y_combine,
                                                       test_size=0.11, random_state=42)
    y_train = np.array(y_train).astype(np.float32)
    y_valid = np.array(y_valid).astype(np.float32)
    y_test = np.array(y_test4).astype(np.float32)        
    
    com = {'y_train':y_train,'y_valid':y_valid,'y_test':y_test,'y':np.array(conservative),
           'train_ind':train_ind,'valid_ind':valid_ind,'test_ind':test_ind}
    
    return com
    
def write_y_index_4CV(conservative):
    ### split test, train, and valid set for 4-fold validation
    ### output (1) CV1 - CV4: train, valid, and test set for four folds.
    indices = np.arange(len(conservative))
    y_rest, y_test1, rest_ind, test_ind1 = train_test_split(conservative, indices, stratify = conservative,
                                                       test_size=0.25, random_state=42) 
    y_rest2, y_test2, rest_ind2, test_ind2 = train_test_split(y_rest, rest_ind, stratify = y_rest,
                                                       test_size=1/3, random_state=42) 
    y_test3, y_test4, test_ind3, test_ind4 = train_test_split(y_rest2, rest_ind2, stratify = y_rest2,
                                                       test_size=1/2, random_state=42) 
    CV1 = make_test_train_valid_for4CV(y_test1,y_test2,y_test3,y_test4,test_ind1,test_ind2,test_ind3,test_ind4,conservative)    
    CV2 = make_test_train_valid_for4CV(y_test1,y_test2,y_test4,y_test3,test_ind1,test_ind2,test_ind4,test_ind3,conservative)
    CV3 = make_test_train_valid_for4CV(y_test1,y_test3,y_test4,y_test2,test_ind1,test_ind3,test_ind4,test_ind2,conservative)    
    CV4 = make_test_train_valid_for4CV(y_test2,y_test3,y_test4,y_test1,test_ind2,test_ind3,test_ind4,test_ind1,conservative)
    # write dataset to pkl
    pickle.dump({'cv1':CV1,'cv2':CV2,'cv3':CV3,'cv4':CV4}, open('index_and_y_conti_4cv.pkl', 'wb')) 
    return CV1,CV2,CV3,CV4

def normalize_tensor(data_tensor):
    ### normalize dataset  
    data_tensor-=np.mean(data_tensor)
    data_tensor/=np.max(np.abs(data_tensor))
    return data_tensor    
     
def write_feature_data_4CV(X,task,CV1,CV2,CV3,CV4):
     
    ### combine all folds in one dictionary after normalization
    data_tensor=np.zeros((X.shape[0],1,X.shape[1],X.shape[2]))
    data_tensor[:,0,:,:] = normalize_tensor(X[:,:,:])
    x_train1 = data_tensor[CV1['train_ind']].astype(np.float32)
    x_test1  = data_tensor[CV1['test_ind']].astype(np.float32)
    x_valid1 = data_tensor[CV1['valid_ind']].astype(np.float32)

    x_train2 = data_tensor[CV2['train_ind']].astype(np.float32)
    x_test2  = data_tensor[CV2['test_ind']].astype(np.float32)
    x_valid2 = data_tensor[CV2['valid_ind']].astype(np.float32)

    x_train3 = data_tensor[CV3['train_ind']].astype(np.float32)
    x_test3  = data_tensor[CV3['test_ind']].astype(np.float32)
    x_valid3 = data_tensor[CV3['valid_ind']].astype(np.float32)

    x_train4 = data_tensor[CV4['train_ind']].astype(np.float32)
    x_test4  = data_tensor[CV4['test_ind']].astype(np.float32)
    x_valid4 = data_tensor[CV4['valid_ind']].astype(np.float32)   
    combine = {'x_train1':x_train1,'x_test1':x_test1,'x_valid1':x_valid1,
                 'x_train2':x_train2,'x_test2':x_test2,'x_valid2':x_valid2,
                 'x_train3':x_train3,'x_test3':x_test3,'x_valid3':x_valid3,
                 'x_train4':x_train4,'x_test4':x_test4,'x_valid4':x_valid4}
    pickle.dump(combine, open(task+'_4cv.pkl', 'wb')) 
    return combine 
  

 



