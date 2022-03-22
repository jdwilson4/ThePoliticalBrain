#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:44:30 2021

@author: seoeunyang
"""
import os, sys,pickle
import argparse
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import caffe
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..'))) # To import ann4brains.
from ann4brains.synthetic.injury import ConnectomeInjury
from ann4brains.nets_new import BrainNetCNN
from ann4brains.utils.metrics import regression_metrics 
from sklearn.metrics import classification_report 

def get_default_hyper_params(net_name,max_iter,base_learning_rate,learning_momentum,weight_decay,mini_batch_size,dropout,x_train):
    
    """Return a dict of the default neural network hyper-parameters""" 
    pars = {}
    pars['net_name'] = net_name  # A unique name of the model.
    pars['dl_framework'] = 'caffe'  # To use different backend neural network frameworks (only caffe for now).
    #
    # Solver parameters
    pars['train_interval'] = 1000  # Display the loss over the training data after 100 iterations.
    pars['test_interval'] = 1000  # Check the model over the test/validation data after this many iterations.
    pars['max_iter'] = max_iter  # Max number of iterations to train the model for.
    pars['snapshot'] = 1000  # After how many iterations should we save the model.
    pars['base_learning_rate'] = base_learning_rate  # Initial learning rate.
    pars['step_size'] = 100000  # After how many iterations should we decrease the learning rate.
    pars['learning_momentum'] = learning_momentum  # Momentum used in learning.
    pars['weight_decay'] = weight_decay  # Weight decay penalty.
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
    pars['dir_snapshots'] = './snapshot'  # Where to store the trained models
    pars['dir_caffe_proto'] = './proto'  # Where to store the caffe prototxt files.
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


def create_pred_across_tasks():
    tasks = ['Affect','Encoding','Retrieval','Reward','WorkingMem','ToM','GoNogo','Empathy','Resting'] 
    hyper = pickle.load(open('hyperparams_conti_268_4cv_hs.pkl', 'rb'))
    data2 = pickle.load(open('index_and_y_conti_4cv.pkl', 'rb'))

    cvs_across_tasks={}
    for w in tasks:
        cvs_all = {}
        for h in range(4):
            data = pickle.load(open(w+'_4cv.pkl', 'rb'))
            all_hyper = hyper[h]
            NET_NAME = w+'_conti_268_cv'+str(h+1)
            conservative = data2['cv'+str(h+1)]['y']
            x_train = data['x_train'+str(h+1)]
            x_valid = data['x_valid'+str(h+1)]
            x_test = data['x_test'+str(h+1)]
            y_train = conservative[data2['cv'+str(h+1)]['train_ind']].astype(np.float32)
            y_valid = conservative[data2['cv'+str(h+1)]['valid_ind']].astype(np.float32)
            y_test = conservative[data2['cv'+str(h+1)]['test_ind']].astype(np.float32)
            pars, archs = get_default_hyper_params(NET_NAME,
                                                   int(all_hyper.loc['max_iters',w]),
                                                   all_hyper.loc['base_learning_rates',w],
                                                   all_hyper.loc['learning_momentums',w],
                                                   all_hyper.loc['weight_decays',w],
                                                   int(all_hyper.loc['mini_batch_sizes',w]),
                                                   all_hyper.loc['dropouts',w],
                                                   x_train)
            NET = BrainNetCNN(NET_NAME, pars, archs)
            NET.fit(x_train, y_train, x_valid, y_valid)
            preds = NET.predict(x_test) # Predict labels of test data
            cor_vals = pearsonr(preds, y_test)
            cvs_all.update({'cv'+str(h+1):{'correlations':cor_vals[0],'ps':cor_vals[1],'predictions':preds,'y':y_test}})
        cvs_across_tasks.update({w:cvs_all})

    pickle.dump(cvs_across_tasks, open('predict_across_tasks_conti_268.pkl', 'wb'))




