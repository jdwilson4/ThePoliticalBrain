#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:45:11 2021

@author: seoeunyang
"""

import os, sys, pickle
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import caffe
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..'))) # To import ann4brains.
from ann4brains.synthetic.injury import ConnectomeInjury
from ann4brains.nets_new import BrainNetCNN
from ann4brains.utils.metrics import regression_metrics 
from imblearn.over_sampling import SMOTE
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

def produce_derivatives():
    tasks = ['Affect','Empathy','Encoding','GoNogo','Resting','Retrieval','Reward','ToM','WorkingMem']
    hyper = pickle.load(open('hyperparams_conti_268_4cv_hs.pkl', 'rb'))
    data2 = pickle.load(open('index_and_y_conti_4cv.pkl', 'rb'))
    
    for w in tasks:
        for h in range(4):
            all_hyper = hyper[h]
            Net_Name = w+'_conti_268_cv'+str(h+1)
            data = pickle.load(open(w+'_4cv.pkl', 'rb'))
            x_train2 = data['x_train'+str(h+1)]
            x_test2 = data['x_test'+str(h+1)]
            x_valid2 = data['x_valid'+str(h+1)]
            conservative = data2['cv'+str(h+1)]['y']
            y_train2 = conservative[data2['cv'+str(h+1)]['train_ind']].astype(np.float32)
            y_valid2 = conservative[data2['cv'+str(h+1)]['valid_ind']].astype(np.float32)
            y_test2 = conservative[data2['cv'+str(h+1)]['test_ind']].astype(np.float32)
            x = np.concatenate([x_test2,x_valid2,x_train2],axis=0)
            y = np.concatenate([y_test2,y_valid2,y_train2],axis=0)
            #configuration
            PRETRAINED_MODEL = 'snapshot/' + Net_Name + '_iter_'+str(all_hyper.loc['iterations'+str(h),w])+'.caffemodel'
            CNN_ARCH_FILE = 'proto/saliency_deploy_'+Net_Name+'.prototxt'
            IMAGE_HW_ORIG = x.shape[2]
            NUM_OUTPUT_LABELS = 1
            LABEL_FOR_SALIENCY = 1
            
            # Load the input image with Caffe
            caffe.set_mode_gpu()
            net = caffe.Classifier(CNN_ARCH_FILE, PRETRAINED_MODEL,
                                   image_dims=(IMAGE_HW_ORIG, IMAGE_HW_ORIG))
                
            #calcaulte predicted derivatives of all people
            y_pred=[]
            store=np.zeros((x.shape[0],x.shape[2],x.shape[3]))
            for i in range(0,len(x)):
               input_image = np.transpose(x[i,:,:])
               finalLabelVector = np.zeros((1,NUM_OUTPUT_LABELS))
               finalLabelVector[0,0] = LABEL_FOR_SALIENCY
               caffe.set_mode_cpu()
               y_pred.append(net.predict([input_image],oversample=False)[0][0]);
               backwardpassData = net.backward(**{net.outputs[0]: finalLabelVector})
               delta = backwardpassData['data']
               store[i,:,:]=delta[0][0]
            
            val=pearsonr(y_pred, y)[0]
            
            #adding matrice of all people
            store_sum=store[0]
            for w in range(1,len(store)):
                store_sum += store[w]
            
            #save predicted derivatives of all people in pkl format
            pkl_name = Net_Name + '_derivative_' + w + '_all.pkl'
            pickle.dump(store, open(pkl_name, 'wb'))
            
            #save mean predicted derivatives of all people in csv format
            store_mean=store_sum/len(store)
            np.savetxt(Net_Name + '_derivative_' + w + '_mean.csv', store_mean, delimiter=",")
            print('-----save predicted derivatives of all people(' + Net_Name + ')-----')














































