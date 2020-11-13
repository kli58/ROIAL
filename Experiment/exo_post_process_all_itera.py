# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 16:07:54 2020

@author: amykj
"""




import numpy as np
import scipy.io as io
import sys
from utility import run_individual_GP, gp_prior

# Unpack command line arguments:
for sub in range(1,4):
    if (sub == 1):
        save_folder = 'Results/sub1/'
        print('Subject 1:')
    elif (sub == 2):
        save_folder = 'Results/sub2/'
        print('Subject 2:')
    else:
        save_folder = 'Results/sub3/'
        print('Subject 3:')
        

    feedback_type = ['ord','pref']
    #Load previous data
    data_matrix_file = save_folder + 'data_matrix.npy' # all prefeences
    
    #determine the ordinal threshold algorithm uses
    data = np.load(data_matrix_file,allow_pickle=True).item()
    sampled_idx = data['sampled_idx']
    sampled_multi_idx = data['sampled_multi_idx']
    sampled_flt_idx = data['sampled_flt_idx']
    relab_sampled_idx = data['relab_sampled_idx']
    
    pref_idxs_post = data['pref_idxs_post']
    pref_labels = data['pref_labels']
    ord_labels = data['ord_labels']
    coactive_idxs = data['coactive_label']
    
    posterior_mean_list = data['posterior_mean_list']
    posterior_mean_whole = data['posterior_mean_whole']
    cts = data['cts']
    params = data['params']
    
    if len(sys.argv) >2:
        ls_file = sys.argv[2] 
        inputs = io.loadmat(ls_file)
        print('load new lengthscales')
        params['lengthscales'] = inputs['lengthscales'][0]
        params['GP_noise_var'] = inputs['GP_noise_var'][0]
     
        
    points_to_sample = params['points_to_sample']
    actions = params['actions']
    grid_shape = params['grid_shape']
    subset_sz = params['subset_sz']
    GP_prior_cov, GP_prior_cov_inv = gp_prior(points_to_sample,params['signal_variance'],params['lengthscales'], params['GP_noise_var'])
    io.savemat(save_folder + 'exo'  +'_prior_cov.mat', { 'GP_prior': GP_prior_cov,'GP_cov_inv':GP_prior_cov_inv})
    
    #re-gernerate idx and corresponding data according to the subspace
    data_GP = {}
    num_trials = len(ord_labels)
    posterior_mean_list = []
    for tr in range(2,num_trials+1):
        print('Updating posterior for Iteration: ', tr,' out of ', num_trials)
        data_GP['pref_labels'] = np.array(pref_labels)[:,1][0:tr-1]
        data_GP['pref_data_idxs']= np.array(pref_idxs_post)[0:tr-1]
        data_GP['ord_data_idxs'] = np.array(sampled_flt_idx)[0:tr]
        data_GP['ord_labels'] = np.array(ord_labels)[0:tr]
        data_GP['coactive_idxs']= np.zeros((0,2))
        data_GP['coactive_label'] = np.array([])
        #pass these data points to a GP
        if len(posterior_mean_list):
            init = posterior_mean_list[-1]
        else:
            init = []
        params['curr_trials'] = len(sampled_flt_idx)
        _, posterior_mean = run_individual_GP(feedback_type,params,GP_prior_cov,GP_prior_cov_inv,data_GP,init,'',post_process = True)
        
        posterior_mean_list.append(posterior_mean)
    
    io.savemat(save_folder + 'exo_post_process_all_iter.mat', { 'posterior_mean_whole': posterior_mean_list})


