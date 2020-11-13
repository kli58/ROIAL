#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:47:56 2020

@author: amyli
"""
import numpy as np
import scipy.io as io
from ROIAL_simulation import run_individual_GP, gp_kernel, init_points_to_sample


def eval_posterior(save_folder,params,state_dim,num_trials,num_runs, feedback_type,path ='',step = 1,load = True,spec_trial_only = False):
    for (run,r) in zip(num_runs,range(len(num_runs))): 
        print('postprocessing run ' + str(run))
        posterior_mean_list = []
        filename = save_folder + 'dim' + str(state_dim) + '_run_' \
                            + str(run)  +'.mat'

        objective_val = io.loadmat(filename)['norm_data']
        grid_shape = np.array(objective_val.shape)
        points_to_sample = init_points_to_sample(grid_shape, state_dim)
   
        if load and path: #load the covariance matrix from the given path
            cov_file = path
            GP_prior_cov = io.loadmat(cov_file)['GP_prior']
            GP_prior_cov_inv = io.loadmat(cov_file)['GP_cov_inv']
        else:
            GP_prior_cov, GP_prior_cov_inv = gp_kernel(points_to_sample,params['signal_variance'],params['lengthscales'], params['GP_noise_var'])
            io.savemat(save_folder + 'dim' + str(state_dim) + '_run_' + str(run)  +'_prior_cov_sig_' +str(params['signal_variance']) +'.mat', { 'GP_prior': GP_prior_cov,'GP_cov_inv':GP_prior_cov_inv})
        print('initiate cov matrix')
       
        params['query_type'] = 'post_process' 
        
        num_list = list(range(1,num_trials+1,step))
        if spec_trial_only:
            num_list = [num_trials]
        for tr in num_list:
            print(tr)
            ord_labels = io.loadmat(filename)['ord_labels'].flatten()[0:tr]
            samp_list = io.loadmat(filename)['sampled_flt_idx'].flatten()[0:tr]
            ord_idxs = samp_list
            if 'pref' in filename and tr > 1:
                pref_idxs = io.loadmat(filename)['pref_idx_post'][0:tr-1]
                pref_labels = io.loadmat(filename)['pref_labels'][:,1][0:tr-1]
            else:
                pref_idxs = np.zeros((0,2))
                pref_labels = np.array([])
      
            #re-gernerate idx and corresponding data according to the subspace
            data = {}
            data['pref_labels'] = np.array(pref_labels)
            data['pref_data_idxs']= np.array(pref_idxs)
            data['ord_data_idxs'] = np.array(ord_idxs)
            data['ord_labels'] = np.array(ord_labels)    
            
            #pass these data points to a GP
            params['curr_trials'] = len(samp_list)
            if len(posterior_mean_list):
                init = posterior_mean_list[-1]
            else:
                init = []
            _, posterior_mean = run_individual_GP(feedback_type,params,GP_prior_cov,GP_prior_cov_inv,data,init,logger = [])
            posterior_mean_list.append(posterior_mean)
        
        io.savemat(save_folder + 'dim' + str(state_dim) + '_run_' + str(run)  +'_post_process.mat', { 'posterior_mean': posterior_mean_list})
    return
    