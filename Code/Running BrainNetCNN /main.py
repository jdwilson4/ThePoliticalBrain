#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:44:30 2021

@author: Seo Eun Yang

This code consists of four parts:
(1) preprocessing step for making a set of train & test data of each task for 4-fold cross validation
(2) finding an optimal set of hyperparameter
(3) running CNNs for predicting political ideology through BrainNetCNN
(4) computing the partial derivatives of the outputs of an CNN with respect to the input features.

To run this code, you should first install caffe and ann4brains under gpu.
We modified the original BrainNetCNN and deconvolutional network architecture for our own purpose.
The original BrainNetCNN code is originally from https://github.com/jeremykawahara/ann4brains
The original deconvolutional network architecture code is originally from
(i) https://shengshuyang.github.io/A-step-by-step-guide-to-Caffe.html
(ii) https://github.com/sukritshankar/CNN-Saliency-Map/blob/master/find_saliency_map.py


""" 
from data_preprocessing import * 
from find_hyperparameter import *
from run import *
from make_derivative import *

# 1. Create Dataset for 4 folds cross validation
conservative, matched, cor_mat_all = draw_data()
CV1,CV2,CV3,CV4 = write_y_index_4CV(conservative)
tasks = list(cor_mat_all.keys())
for w in tasks:
    _ = write_feature_data_4CV(cor_mat_all[w],w,CV1,CV2,CV3,CV4)

# 2. Find optimal hyperparameter set for each fold per task
write_optimal_hyperparameter()

# 3. Using selected hyperparameter, run the model and save predicted values
create_pred_across_tasks()

# 4. Deconvolutional NN for creating partial derivative matrix
produce_derivatives()



