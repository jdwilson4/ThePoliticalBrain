"""
Created on Sun Dec 30 13:22:54 2024

@author: Seo Eun Yang (se.yang@northeastern.edu)

This code consists of three parts:
(1) preprocessing step for making a set of train, validation & test data of each task for 4-fold cross validation
    - Split the Data: Divide our dataset into three parts: training, validation, and test datasets
(2) finding an optimal set of hyperparameter via Gridsearch method
    - Train the Model: Train the model using the training dataset with different sets of hyperparameters.
    - Validate the Model: After training the model with each set of hyperparameters, use the validation dataset to predict the target values.
    - Compute Validation Metric: Compute the Pearson correlation between the predicted values and the actual target values from the validation dataset.
    - Select Best Hyperparameters: Identify the set of hyperparameters that produces the highest Pearson correlation on the validation dataset.
(3) running CNNs for predicting political ideology through BrainNetCNN 
    - After selecting the best hyperparameters, train the model on the combined training and validation dataset.
    - Finally, evaluate the model on the test dataset, which provides an out-of-sample prediction and serves as a final check on the model's performance.

To run this code, you should first install Caffe and ann4brains under GPU.
python2.7-conda5.2 were used.
We modified our origianl python scripts (2021 version) and BrainNetCNN models for our own purpose.
The original BrainNetCNN code is originally from https://github.com/jeremykawahara/ann4brains 


""" 
from data_preprocessing import * 
from find_optimal_hyperparameters import *
from get_predicted_outcomes import * 

# 1. Create Training, Validation, and Test Dataset for 4 folds cross validation
conservative, matched, cor_mat_all = draw_data()
CV1,CV2,CV3,CV4 = write_y_index_4CV(conservative)
tasks = list(cor_mat_all.keys())
for w in tasks:
    _ = write_feature_data_4CV(cor_mat_all[w],w,CV1,CV2,CV3,CV4)

# 2. Find optimal hyperparameter set for each fold per task
store_performances_from_combination_of_hyperparameter() # Run models for all possible combinations of hyperparameters for each task and fold
find_best_hyperparameter_per_fold_from_gridsearch() # hyperparams_conti_268_4cv_hs_complete.pkl will be created

# 3. Using selected hyperparameter, run the model and save predicted values
create_pred_across_tasks() # predict_across_tasks_conti_268.pkl will be created