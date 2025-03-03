#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 23:03:56 2021

@author: seoeunyang
"""
from nilearn import plotting
from matplotlib.pyplot import cm
import os, sys, pickle
import numpy as np
import pandas as pd
import itertools
import seaborn as sns
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
os.makedirs('brain_plot')

def find_right_loc(k1):
    loc1 = np.array(AAL[brain_loc[k1]]) 
    if brain_loc[k1] == 'Frontal7':
        loc1 = np.array([228])
    if sum(loc1>252)>0:
        loc1[loc1<252] = loc1[loc1<252] - 1
        loc1[loc1>252] = loc1[loc1>252] - 2
    else:
        loc1 = loc1 - 1   
    return loc1
     
def magnitude_compare_conti():
    training_all={}
    for i in range(len(tasks)):  
        for h in range(4):
            if h==0:
                mean_deri=pd.read_csv('csv/'+tasks[i]+'_conti_268_cv'+str(h+1)+'_derivative_'+tasks[i]+'_mean.csv', sep=",",header=None)
            mean_deri += pd.read_csv('csv/'+tasks[i]+'_conti_268_cv'+str(h+1)+'_derivative_'+tasks[i]+'_mean.csv', sep=",",header=None)
        mean_deri /= 4
        mean_deri=mean_deri.to_numpy()  
        mat78 = np.zeros((len(AAL),len(AAL)))
        for k1 in range(len(AAL)):
            for k2 in range(len(AAL)):
                loc1 = find_right_loc(k1)
                loc2 = find_right_loc(k2)
                selected = mean_deri[loc1,:] 
                mat78[k1,k2] = selected[:,loc2].mean() 
        mat=pd.DataFrame(mat78)  
        upper=mat.where(np.triu(np.ones(mat.shape),k=1).astype(np.bool))
        lower=(mat.T).where(np.triu(np.ones(mat.shape),k=1).astype(np.bool))
        df = (upper+lower)/2
        df = df.stack().reset_index()
        df.columns = ['Row','Column','Value'] 
        df['magnitude']=abs(df['Value'])
        row_name, col_name= [],[]
        for w1 in range(len(df)):
            row_name.append(brain_loc[df.loc[w1,'Row']])
            col_name.append(brain_loc[df.loc[w1,'Column']])
        df['rowname']=row_name
        df['colname']=col_name  
        abs_mag = []
        for j in range(78):
            node = pd.concat([df.loc[df['Row']==j],df.loc[df['Column']==j]])
            abs_mag.append(abs(node['Value']).sum())  
        training_all.update({tasks[i]:np.array(abs_mag)})
    training_all.update({'broad2':brain_loc})
    training_all2 = pd.DataFrame(training_all)
    training_all2.index=brain_loc
    combine = pd.merge(training_all2,label[['broad2','abb_x','AAL_new','x','y','z']],on='broad2')
    return combine

tasks=['Affect','Empathy','Encoding','GoNogo','Resting','Retrieval','Reward','ToM','WorkingMem']
label=pickle.load(open('label.pkl','rb'))
brain_loc = list(label.broad2)
magnitude = magnitude_compare_conti()  

## plot Empathy, Reward, and Retrieval
min_val2 = magnitude[['Empathy','Reward','Retrieval']].min().min()
max_val2 = magnitude[['Empathy','Reward','Retrieval']].max().max()
Empathy_val2 = magnitude[['Empathy','x','y','z','abb_x']].sort_values('Empathy',ascending=False) 
Reward_val2 = magnitude[['Reward','x','y','z','abb_x']].sort_values('Reward',ascending=False) 
Retrieval_val2 = magnitude[['Retrieval','x','y','z','abb_x']].sort_values('Retrieval',ascending=False)
Empathy_val = Empathy_val2[:20] #top 20
Reward_val = Reward_val2[:20]  #top 20
Retrieval_val = Retrieval_val2[:20]  #top 20

min_val = min(Empathy_val.Empathy.min(),Reward_val.Reward.min(),Retrieval_val.Retrieval.min())
max_val = max(Empathy_val.Empathy.max(),Reward_val.Reward.max(),Retrieval_val.Retrieval.max())
thre_val = 0.06 

# Draw plots
plotting.plot_markers( 
    Empathy_val2['Empathy'], 
    np.array(Empathy_val2[['x','y','z']]),
    title='Empathy',
    node_cmap=cm.YlOrRd,
    display_mode='lyrz',
    node_threshold=thre_val,
    #node_size=20, 
    node_vmin=min_val2,
    node_vmax=max_val2
)  
plt.savefig('brain_plot/empathy_node.pdf', dpi=400) 
 
plotting.plot_markers( 
    Reward_val2['Reward'], 
    np.array(Reward_val2[['x','y','z']]),
    title='Reward',
    node_cmap=cm.YlOrRd,
    display_mode='lyrz',
    node_threshold=thre_val,
    node_vmin=min_val2,
    node_vmax=max_val2
) 
plt.savefig('brain_plot/reward_node.pdf', dpi=400) 

plotting.plot_markers( 
    Retrieval_val2['Retrieval'], 
    np.array(Retrieval_val2[['x','y','z']]),
    title='Retrieval',
    node_cmap=cm.YlOrRd,
    display_mode='lyrz',
    node_threshold=thre_val,
    node_vmin=min_val2,
    node_vmax=max_val2
) 
plt.savefig('brain_plot/retrieval_node.pdf', dpi=400) 
 
 
