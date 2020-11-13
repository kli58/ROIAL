#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:48:19 2020

@author: amyli
"""


#relabel preference and ordinal feedback idx
import scipy.io as io
import numpy as np
from utility import gp_prior, run_individual_GP, init_points_to_sample

save_folder = '/Users/amyli/Desktop/CalTech/gitrepo/exo_results/Results/sub1/'

#load experiment data
data_matrix_file = save_folder + 'data_matrix.npy'
input_file = save_folder + 'active_GP_params.mat'
inputs = io.loadmat(input_file)

data = np.load(data_matrix_file,allow_pickle=True).item()
sampled_idx = data['sampled_idx']
sampled_multi_idx = data['sampled_multi_idx']
sampled_flt_idx = data['sampled_flt_idx']
pref_idxs = data['pref_idxs']
pref_idxs_post = data['pref_idxs_post']
pref_labels = data['pref_labels']
ord_idxs = data['ord_idxs']
ord_labels = data['ord_labels']
current_action = data['returned_action']
relab_sampled_idx = data['relab_sampled_idx']
params = data['params']
validation_actions = data['validation_actions']

lbs = inputs['lower_bounds'][0]
ubs = inputs['upper_bounds'][0]
state_dim = inputs['dims'][0][0]

#initialize finer grid
new_grid_shape = np.array([20,20,20,10])
new_points = init_points_to_sample(new_grid_shape, '', lbs, ubs, state_dim)

#%%
#load points sampled
visited_actions = params['points_to_sample'][sampled_flt_idx]
new_visited_actions = visited_actions.copy()

# for i in range(len(new_points)):
#     if list(new_points[i,:]) not in visited_actions.tolist():
#         new_visited_actions = np.append(new_visited_actions,[new_points[i,:]],axis = 0)

feedback_type = ['ord','pref']
data_GP = {}
if len(pref_labels):
    data_GP['pref_labels'] = np.array(pref_labels)[:,1]
    data_GP['pref_data_idxs']= np.array(pref_idxs)
else:
    data_GP['pref_labels'] = np.array([])
    data_GP['pref_data_idxs']= np.zeros((0,2))
data_GP['ord_data_idxs'] = np.array(ord_idxs)
data_GP['ord_labels'] = np.array(ord_labels)
data_GP['coactive_idxs']= np.zeros((0,2))
data_GP['coactive_label'] = np.array([])

sub_points = np.append(visited_actions,new_points,axis = 0)

#initialize covariance matrix
GP_prior_cov, GP_prior_cov_inv = gp_prior(sub_points,params['signal_variance'],params['lengthscales'], params['GP_noise_var'])

#evaluate posterior
_, posterior_mean = run_individual_GP(feedback_type,params,GP_prior_cov,GP_prior_cov_inv,data_GP,[],'',post_process = True)

#save results to a mat file
io.savemat(save_folder + 'exo_finer_posterior.mat', {'new_points':new_visited_actions, 'posterior_mean_whole': posterior_mean})

