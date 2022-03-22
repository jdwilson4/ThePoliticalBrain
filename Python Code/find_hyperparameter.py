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
import scipy.stats
import itertools
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import caffe
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..'))) # To import ann4brains.
from ann4brains.synthetic.injury import ConnectomeInjury
from ann4brains.nets_new import BrainNetCNN
from ann4brains.utils.metrics import regression_metrics 
from sklearn.metrics import classification_report 
import random

random.seed(1234)

def get_default_hyper_params(net_name,max_iter,base_learning_rate,learning_momentum,weight_decay,mini_batch_size,dropout,x_train):
    #
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

def create_corr_index(w,name,h):
    hyp_set={'max_iters':np.array([10000]*10)*np.array(range(1,11)),
    'base_learning_rates':np.array([1e-2]*9)*np.array(range(1,10)),
    'learning_momentums':np.array([0.1]*9)*np.array(range(1,10)),
    'weight_decays':np.array([0.0001]*10)*np.array(range(1,11)),
    'mini_batch_sizes':[10,11,12,13,14,15],
    'dropouts':np.array([0.1]*9)*np.array(range(1,10))} 
    corrs={}
    val_list=hyp_set[name]
    data = pickle.load(open(w+'_conti_268_cv'+str(h+1)+'_'+name+'_results_'+str(val_list[len(val_list)-1])+'.pkl', 'rb'))
    corrs.update({w:data['corr']})
    results = pd.DataFrame(corrs)
    try:
        results.index=data[name]
    except ValueError:
        results.index=data['val']
    except KeyError:
        results.index=data['val']
    hyperparam = (results.idxmax(axis=0)).to_dict()
    return corrs,hyperparam

# load dataset  
def run_all_combination(w,h):
    ### create prediction results from some possible combination of hyperparameters sequentially.
    ### input (1) w : task
    ###       (2) h : fold
    data = pickle.load(open(w +'_4cv.pkl', 'rb'))
    data2 = pickle.load(open('index_and_y_conti_4cv.pkl', 'rb'))
    cvs = ['cv1','cv2','cv3','cv4']
    
    # lists of hyperparameter candidates
    max_iters = np.array([10000]*10)*np.array(range(1,11))
    base_learning_rates = np.array([1e-2]*9)*np.array(range(1,10))
    learning_momentums = np.array([0.1]*9)*np.array(range(1,10))
    weight_decays = np.array([0.0001]*10)*np.array(range(1,11))
    mini_batch_sizes = [10,11,12,13,14,15]
    dropouts = np.array([0.1]*9)*np.array(range(1,10))         

    # max_iters
    corr_h,pred_h,y_tests = [],[],[]
    for i in range(len(max_iters)):
        val = max_iters[i]
        net_name = w+'_conti_268_'+cvs[h]
        conservative = data2[cvs[h]]['y']
        x_train = data['x_train'+str(h+1)]
        x_valid = data['x_valid'+str(h+1)]
        x_test = data['x_test'+str(h+1)]
        y_train = conservative[data2[cvs[h]]['train_ind']].astype(np.float32)
        y_valid = conservative[data2[cvs[h]]['valid_ind']].astype(np.float32)
        y_test = conservative[data2[cvs[h]]['test_ind']].astype(np.float32)
        pars, archs = get_default_hyper_params(net_name,val,0.01,0.9,0.0005,14,0.5,x_train)
        NET = BrainNetCNN(net_name, pars, archs) # Create BrainNetCNN model
        NET.fit(x_train, y_train, x_valid, y_valid) # Train (regress only on class 0)
        preds = NET.predict(x_test) # Predict labels of test data
        pred_h.append(preds)
        y_tests.append(y_test)
        corr_h.append(pearsonr(preds, y_test)[0])
        pickle.dump({'max_iters':max_iters,'corr':corr_h,'pred':pred_h, 'y_test':y_tests}, open(net_name+'_max_iters_results_'+str(val)+'.pkl', 'wb'))
    
    # dropout
    corrs1,iterations = create_corr_index(w,'max_iters',h)
    corr_h,pred_h,y_tests = [],[],[]
    for i in range(len(dropouts)):
        val = dropouts[i]
        net_name = w+'_conti_268_'+cvs[h]
        conservative = data2[cvs[h]]['y']
        x_train = data['x_train'+str(h+1)]
        x_valid = data['x_valid'+str(h+1)]
        x_test = data['x_test'+str(h+1)]
        y_train = conservative[data2[cvs[h]]['train_ind']].astype(np.float32)
        y_valid = conservative[data2[cvs[h]]['valid_ind']].astype(np.float32)
        y_test = conservative[data2[cvs[h]]['test_ind']].astype(np.float32)
        pars, archs = get_default_hyper_params(net_name,iterations[w],0.01,0.9,0.0005,14,val,x_train)
        NET = BrainNetCNN(net_name, pars, archs) # Create BrainNetCNN model
        NET.fit(x_train, y_train, x_valid, y_valid) # Train (regress only on class 0)
        preds = NET.predict(x_test) # Predict labels of test data
        pred_h.append(preds)
        y_tests.append(y_test)
        corr_h.append(pearsonr(preds, y_test)[0])
        pickle.dump({'dropouts':dropouts,'corr':corr_h,'pred':pred_h, 'y_test':y_tests}, open(net_name+'_dropouts_results_'+str(val)+'.pkl', 'wb'))
    
    # base_learning_rates
    corrs1,iterations = create_corr_index(w,'max_iters',h)
    corrs2,dropouts = create_corr_index(w,'dropouts',h)
    corr_h,pred_h,y_tests = [],[],[]
    for i in range(len(base_learning_rates)):
        val = base_learning_rates[i]
        net_name = w+'_conti_268_'+cvs[h]
        conservative = data2[cvs[h]]['y']
        x_train = data['x_train'+str(h+1)]
        x_valid = data['x_valid'+str(h+1)]
        x_test = data['x_test'+str(h+1)]
        y_train = conservative[data2[cvs[h]]['train_ind']].astype(np.float32)
        y_valid = conservative[data2[cvs[h]]['valid_ind']].astype(np.float32)
        y_test = conservative[data2[cvs[h]]['test_ind']].astype(np.float32)
        pars, archs = get_default_hyper_params(net_name,iterations[w],val,0.9,0.0005,14,dropouts[w],x_train)
        NET = BrainNetCNN(net_name, pars, archs) # Create BrainNetCNN model
        NET.fit(x_train, y_train, x_valid, y_valid) # Train (regress only on class 0)
        preds = NET.predict(x_test) # Predict labels of test data
        pred_h.append(preds)
        y_tests.append(y_test)
        corr_h.append(pearsonr(preds, y_test)[0])
        pickle.dump({'base_learning_rates':base_learning_rates,'corr':corr_h,'pred':pred_h, 'y_test':y_tests}, open(net_name+'_base_learning_rates_results_'+str(val)+'.pkl', 'wb'))

    # learning_momentums
    corrs1,iterations = create_corr_index(w,'max_iters',h)
    corrs2,dropouts = create_corr_index(w,'dropouts',h)
    corrs3,base_learning_rates = create_corr_index(w,'base_learning_rates',h)
    corr_h,pred_h,y_tests = [],[],[]
    for i in range(len(learning_momentums)):
        val=learning_momentums[i]
        net_name = w+'_conti_268_'+cvs[h]
        conservative = data2[cvs[h]]['y']
        x_train = data['x_train'+str(h+1)]
        x_valid = data['x_valid'+str(h+1)]
        x_test = data['x_test'+str(h+1)]
        y_train = conservative[data2[cvs[h]]['train_ind']].astype(np.float32)
        y_valid = conservative[data2[cvs[h]]['valid_ind']].astype(np.float32)
        y_test = conservative[data2[cvs[h]]['test_ind']].astype(np.float32)
        pars, archs = get_default_hyper_params(net_name,iterations[w],base_learning_rates[w],val,0.0005,14,dropouts[w],x_train)
        NET = BrainNetCNN(net_name, pars, archs) # Create BrainNetCNN model
        NET.fit(x_train, y_train, x_valid, y_valid) # Train (regress only on class 0)
        preds = NET.predict(x_test) # Predict labels of test data
        pred_h.append(preds)
        y_tests.append(y_test)
        corr_h.append(pearsonr(preds, y_test)[0])
        pickle.dump({'learning_momentums':learning_momentums,'corr':corr_h,'pred':pred_h, 'y_test':y_tests}, open(net_name+'_learning_momentums_results_'+str(val)+'.pkl', 'wb'))

    # weight_decays
    corrs1,iterations = create_corr_index(w,'max_iters',h)
    corrs2,dropouts = create_corr_index(w,'dropouts',h)
    corrs3,base_learning_rates = create_corr_index(w,'base_learning_rates',h)
    corrs4,learning_momentums = create_corr_index(w,'learning_momentums',h)
    corr_h,pred_h,y_tests = [],[],[]
    for i in range(len(weight_decays)):
        val=weight_decays[i]
        net_name = w+'_conti_268_'+cvs[h]
        conservative = data2[cvs[h]]['y']
        x_train = data['x_train'+str(h+1)]
        x_valid = data['x_valid'+str(h+1)]
        x_test = data['x_test'+str(h+1)]
        y_train = conservative[data2[cvs[h]]['train_ind']].astype(np.float32)
        y_valid = conservative[data2[cvs[h]]['valid_ind']].astype(np.float32)
        y_test = conservative[data2[cvs[h]]['test_ind']].astype(np.float32)
        pars, archs = get_default_hyper_params(net_name,iterations[w],base_learning_rates[w],learning_momentums[w],val,14,dropouts[w],x_train)
        NET = BrainNetCNN(net_name, pars, archs) # Create BrainNetCNN model
        NET.fit(x_train, y_train, x_valid, y_valid) # Train (regress only on class 0)
        preds = NET.predict(x_test) # Predict labels of test data
        pred_h.append(preds)
        y_tests.append(y_test)
        corr_h.append(pearsonr(preds, y_test)[0])
        pickle.dump({'weight_decays':weight_decays,'corr':corr_h,'pred':pred_h, 'y_test':y_tests}, open(net_name+'_weight_decays_results_'+str(val)+'.pkl', 'wb'))

    # mini_batch_sizes
    corrs1,iterations = create_corr_index(w,'max_iters',h)
    corrs2,dropouts = create_corr_index(w,'dropouts',h)
    corrs3,base_learning_rates = create_corr_index(w,'base_learning_rates',h)
    corrs4,learning_momentums = create_corr_index(w,'learning_momentums',h)
    corrs5,weight_decays = create_corr_index(w,'weight_decays',h)
    corr_h,pred_h,y_tests = [],[],[]
    for i in range(len(mini_batch_sizes)):
        val=mini_batch_sizes[i]
        net_name = w+'_conti_268_'+cvs[h]
        conservative = data2[cvs[h]]['y']
        x_train = data['x_train'+str(h+1)]
        x_valid = data['x_valid'+str(h+1)]
        x_test = data['x_test'+str(h+1)]
        y_train = conservative[data2[cvs[h]]['train_ind']].astype(np.float32)
        y_valid = conservative[data2[cvs[h]]['valid_ind']].astype(np.float32)
        y_test = conservative[data2[cvs[h]]['test_ind']].astype(np.float32)
        pars, archs = get_default_hyper_params(net_name,iterations[w],base_learning_rates[w],learning_momentums[w],weight_decays[w],val,dropouts[w],x_train)
        NET = BrainNetCNN(net_name, pars, archs) # Create BrainNetCNN model
        NET.fit(x_train, y_train, x_valid, y_valid) # Train (regress only on class 0)
        preds = NET.predict(x_test) # Predict labels of test data
        pred_h.append(preds)
        y_tests.append(y_test)
        corr_h.append(pearsonr(preds, y_test)[0])
        pickle.dump({'mini_batch_sizes':mini_batch_sizes,'corr':corr_h,'pred':pred_h, 'y_test':y_tests}, open(net_name+'_mini_batch_sizes_results_'+str(val)+'.pkl', 'wb'))

def compare_results(h,name,task,real_val):
    values=real_val[name]
    output = pickle.load(open(task+'_conti_268_cv'+str(h)+'_'+name+'_results_'+str(values[-1])+'.pkl', 'rb'))
    return output

def find_loc_max(value_1):
    loca1 = np.where(value_1 == np.nanmax(value_1))[0][0]
    return loca1

def find_optimal_hyperparameter(w,h,real_val,names):
    all_values,all_maxes,all_hyper={},{},{}
    value_1=compare_results(h+1,'max_iters',w,real_val)
    value_2=compare_results(h+1,'dropouts',w,real_val)
    value_3=compare_results(h+1,'base_learning_rates',w,real_val)
    value_4=compare_results(h+1,'learning_momentums',w,real_val)
    value_5=compare_results(h+1,'weight_decays',w,real_val)
    value_6=compare_results(h+1,'mini_batch_sizes',w,real_val)
    hyper_found = [real_val['max_iters'][find_loc_max(value_1)],
                 real_val['dropouts'][find_loc_max(value_2)],
                 real_val['base_learning_rates'][find_loc_max(value_3)],
                 real_val['learning_momentums'][find_loc_max(value_4)],
                 real_val['weight_decays'][find_loc_max(value_5)],
                 real_val['mini_batch_sizes'][find_loc_max(value_6)]]
    return hyper_found

def write_optimal_hyperparameter():
    real_val={'max_iters':np.array([10000]*10)*np.array(range(1,11)),
    'base_learning_rates':np.array([1e-2]*9)*np.array(range(1,10)),
    'learning_momentums':np.array([0.1]*9)*np.array(range(1,10)),
    'weight_decays':np.array([0.0001]*10)*np.array(range(1,11)),
    'mini_batch_sizes':[10,11,12,13,14,15],
    'dropouts':np.array([0.1]*9)*np.array(range(1,10))} #the range of hyperparameter
    names=['max_iters','dropouts','base_learning_rates','learning_momentums','weight_decays','mini_batch_sizes']
    tasks = ['Affect','Encoding','Retrieval','Reward','WorkingMem','ToM','GoNogo','Empathy','Resting']
    all_hyper_hs = {}
    for h in range(4):
        all_hyper = {}
        for w in tasks:
            run_all_combination(w,h)
            hyper = find_optimal_hyperparameter(w,h,real_val,names)
            all_hyper.update({w:hyper})
        all_hypers = pd.DataFrame(all_hyper)
        all_hypers.index = names
        all_hyper_hs.update({h:all_hypers})
    pickle.dump(all_hyper_hs,open('hyperparams_conti_268_4cv_hs.pkl', 'rb'))






