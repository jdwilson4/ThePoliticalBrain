#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  23 22:14:17 2024

@author: Seo Eun Yang (se.yang@northeastern.edu)
"""

import os, sys,pickle,time
import argparse
import numpy as np
import pandas as pd
import scipy.stats
import itertools
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import caffe
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'ann4brains'))) # To import ann4brains.
from ann4brains.synthetic.injury import ConnectomeInjury
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'ann4brains'))) # To import ann4brains.
from ann4brains.nets import BrainNetCNN
from ann4brains.utils.metrics import regression_metrics 
from sklearn.metrics import classification_report 
import random

random.seed(1234)

# list of hyperparameters
names=['dropout','base_learning_rate','mini_batch_size','max_iter']
tasks = ['Affect','Encoding','Retrieval','Reward','WorkingMem','ToM','GoNogo','Empathy','Resting']

# the range of possible values from each hyperparameter (dictionary format) 
real_val={   
    'dropout':[0.3,0.4,0.5,0.6,0.7],
    'base_learning_rate':[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], 
    'mini_batch_size':[8,16,32],
    'max_iter':[50,100,150,200,250]
} 

# the range of possible values from each hyperparameter (list format)
param_list=[ 
    [0.3,0.4,0.5,0.6,0.7], #'dropout'
    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],#'base_learning_rate' 
    [8,16,32], #'mini_batch_size' 
    [50,100,150,200,250] #'max_iter'
]

def get_default_hyper_params(net_name,dropout,base_learning_rate,mini_batch_size,max_iter,x_train):
    #
    """Return a dict of the default neural network hyper-parameters""" 
    pars = {}
    pars['net_name'] = net_name  # A unique name of the model.
    pars['dl_framework'] = 'caffe'  # To use different backend neural network frameworks (only caffe for now).
    #
    # Solver parameters
    pars['train_interval'] = 5  # Display the loss over the training data after 100 iterations.
    pars['test_interval'] = 5  # Check the model over the test/validation data after this many iterations.
    pars['max_iter'] = max_iter  # Max number of iterations to train the model for.
    pars['snapshot'] = 5  # After how many iterations should we save the model.
    pars['base_learning_rate'] = base_learning_rate  # Initial learning rate.
    pars['step_size'] = 100000  # After how many iterations should we decrease the learning rate.
    pars['learning_momentum'] = 0.1  # Momentum used in learning.
    pars['weight_decay'] = 0.0001  # Weight decay penalty.
    pars['loss'] = 'EuclideanLoss'  # The loss to use (currently only EuclideanLoss works)
    pars['compute_metrics_func'] = regression_metrics  # Can override with your own as long as follows format.
    #
    # Network parameters
    pars['train_batch_size'] = mini_batch_size  # Size of the training mini-batch.
    #
    # The number of test samples should equal, test_batch_size * test_iter.
    # Easiest to set pars['test_iter'] to equal the number of test samples pars['test_batch_size'] = 1
    pars['test_batch_size'] = 1  # Size of the testing mini-batch.
    # pars['test_iter'] = 56  # How many test samples we have (we'll set this based on the data)
    pars['ntop'] = 2  # How many outputs for the hdf5 data layer (e.g., 'data', 'label' = 2)
    #
    pars['dir_snapshots'] = 'ann4brains/snapshot'# 'ann4brains/snapshot'  # Where to store the trained models
    pars['dir_caffe_proto'] = 'ann4brains/proto'# 'ann4brains/proto'  # Where to store the caffe prototxt files.
    #
    archs = [ # We specify the architecture like this.
        ['e2n', {'n_filters': 16,  # e2n layer with 16 filters.
                 'kernel_h': x_train.shape[2], 
                 'kernel_w': x_train.shape[3]}], # Same dimensions as spatial inputs. 
        ['dropout', {'dropout_ratio': dropout}], # Dropout at 0.5
        ['relu',    {'negative_slope': 0.33}], # For leaky-ReLU
        ['fc',      {'n_filters': 30}],  # Fully connected (n2g) layer with 30 filters.
        ['relu',    {'negative_slope': 0.33}],
        ['out',     {'n_filters': 1}]]  # Output layer with 1 nodes as output.
    #
    return pars,archs

for w in tasks:
    print(f'--- {w} ---')
    for h in range(4): 
        print(f'--- {h} ---')
        # create a dataframe to record every possible combination of hyperparameter values from the list above?
        all_combinations=pd.DataFrame([element for element in itertools.product(*param_list)])
        all_combinations.columns = ['dropout','base_learning_rate','mini_batch_size','max_iter']
        all_combinations['cor_train']=[0]*len(all_combinations)
        all_combinations['p_train']=[0]*len(all_combinations)
        all_combinations['cor_valid']=[0]*len(all_combinations)
        all_combinations['p_valid']=[0]*len(all_combinations) 
        all_combinations['mae_train']=[0]*len(all_combinations) 
        all_combinations['mae_valid']=[0]*len(all_combinations) 

        param_list = param_list
        hardware='gpu'
        data = pickle.load(open(w +'_4cv.pkl', 'rb'))
        data2 = pickle.load(open('index_and_y_conti_4cv.pkl', 'rb'))
        cvs = ['cv1','cv2','cv3','cv4']
        dir_data = 'ann4brains/data_'+w+'_'+str(h+1)  

        net_name = w+'_conti_268_'+cvs[h] 
        conservative = data2[cvs[h]]['y']
        x_train = data['x_train'+str(h+1)]
        x_valid = data['x_valid'+str(h+1)]  
        y_train = conservative[data2[cvs[h]]['train_ind']].astype(np.float32)
        y_valid = conservative[data2[cvs[h]]['valid_ind']].astype(np.float32)   
        errors=[]
        for i in range(len(all_combinations)):
            try: 
                dropout=all_combinations.loc[i,'dropout']
                base_learning_rate=all_combinations.loc[i,'base_learning_rate']
                mini_batch_size=all_combinations.loc[i,'mini_batch_size']
                max_iter=all_combinations.loc[i,'max_iter'] 

                pars, archs = get_default_hyper_params(net_name,dropout,base_learning_rate,mini_batch_size,max_iter,x_train) 
                NET = BrainNetCNN(net_name, pars, archs, dir_data=dir_data)  
                NET.fit(x_train, y_train, x_valid, y_valid) # Run a model with training and validation dataset
                preds_tr = NET.predict(x_train) 
                preds_va = NET.predict(x_valid) 

                re_train=pearsonr(preds_tr, y_train) 
                re_valid=pearsonr(preds_va, y_valid)
                all_combinations.loc[i,'cor_train']=re_train[0]
                all_combinations.loc[i,'p_train']=np.round(re_train[1],3)
                all_combinations.loc[i,'cor_valid']=re_valid[0]
                all_combinations.loc[i,'p_valid']=np.round(re_valid[1],3) 
                all_combinations.loc[i,'mae_train']=np.mean((abs(preds_tr - y_train)))
                all_combinations.loc[i,'mae_valid']=np.mean((abs(preds_va - y_valid)))  
            except:  
                errors.append(i)

        os.exists(f'hyperparameter_search') or os.makedirs(f'hyperparameter_search')
        pickle.dump(all_combinations,open(f'hyperparameter_search/{net_name}.pkl','wb'))
        pickle.dump(errors,open(f'hyperparameter_search/error_index_{net_name}.pkl','wb'))
        print(f'--- # error {len(errors)}---')


# Combine all hyperparameter combinations and save the optimal hyperparameters
folder='hyperparameter_search'
for task in tasks:
    for fold in range(1,5):
        if (fold==1) and (task=='Affect'):
            task_combi = pd.read_pickle(open(f'{folder}/{task}_conti_268_cv{fold}.pkl','rb'))
            task_combi['fold']=fold
            task_combi['task']=task
        else:
            add_combi = pd.read_pickle(open(f'{folder}/{task}_conti_268_cv{fold}.pkl','rb'))
            add_combi['fold']=fold 
            add_combi['task']=task
            task_combi = pd.concat([task_combi,add_combi],axis=0, ignore_index=True)

hyperparameter_combination_optimal = task_combi.loc[task_combi['cor_train']!=0].groupby(['task','fold']).apply(lambda x: x.nlargest(1, 'cor_valid'))

pickle.dump(task_combi,open('hyperparameter_combination_all.pkl','wb')) 
pickle.dump(hyperparameter_combination_optimal,open('hyperparameter_combination_optimal.pkl','wb'))