#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:29:15 2020

@author: mtucker
"""



import numpy as np
import scipy.io as io

# permutation feature importance with knn for regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot

def get_ordinal_feedback(obj_value,ordinal_threshold):
    for i in range(1,len(ordinal_threshold)):
        if obj_value <= ordinal_threshold[i]:
            obj_label = i
            break
    return obj_label

def getPermutationImportance(save_folder):
    filename = save_folder + 'exo_post_process.mat'
    posterior_mean = io.loadmat(filename)['posterior_mean']
    
    # Load data from experiment
    data_matrix_file = save_folder + 'data_matrix.npy' # all preferences
    data = np.load(data_matrix_file,allow_pickle=True).item()
    params = data['params']
    
    # Get actions associated with posterior mean
    points_to_sample = params['points_to_sample']

    # Determine the ordinal threshold algorithm uses
    ord_thresh = params['ord_threshold_estimate']
    ordinal_threshold_comp = [-np.inf]
    ordinal_threshold_comp.extend(np.cumsum(ord_thresh))
    ordinal_threshold_comp= np.append(ordinal_threshold_comp,np.inf)
    
    # Get ordinal label for each action based on the posterior mean
    labels = []
    posterior_mean_flt = posterior_mean.flatten()
    for i in posterior_mean_flt:
        label = get_ordinal_feedback(i, ordinal_threshold_comp)
        labels.append(label)
        

    #%%
    # define the model
    model = KNeighborsRegressor()
    
    #%%

    # data to use
    X = points_to_sample;
    y = labels;
    
    # fit the model
    model.fit(X, y)
    
    # perform permutation importance
    results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
    
    #%%
    
    # get importance
    importance = results.importances_mean
    
    # summarize feature importance
    for i,v in enumerate(importance):
    	print('Feature: %0d, Score: %.5f' % (i,v))
        
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()
    
    return importance

## Subject 1
save_folder = '../exo_results/Results/sub1/'
print('Subject 1:')
importance =  getPermutationImportance(save_folder)

save_folder = '../exo_results/Results/sub2/'
print('Subject 2:')
importance =  getPermutationImportance(save_folder)

save_folder = '../exo_results/Results/sub3/'
print('Subject 3:')
importance =  getPermutationImportance(save_folder)