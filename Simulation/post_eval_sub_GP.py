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
   
        if load and path:
            cov_file = path
            GP_prior_cov = io.loadmat(cov_file)['GP_prior']
            GP_prior_cov_inv = io.loadmat(cov_file)['GP_cov_inv']
        else:
            GP_prior_cov, GP_prior_cov_inv = gp_kernel(points_to_sample,params['signal_variance'],params['lengthscales'], params['GP_noise_var'])
            io.savemat(save_folder + 'dim' + str(state_dim) + '_run_' + str(run)  +'_prior_cov_sig_' +str(params['signal_variance']) +'.mat', { 'GP_prior': GP_prior_cov,'GP_cov_inv':GP_prior_cov_inv})
        print('initiate cov matrix')
       
        params['query_type'] = 'post_process'
        delta_int = 2/params['num_category']
        ordinal_threshold_estimate = np.array([delta_int]*(params['num_category']-1))
        ordinal_threshold_estimate[0] = -0.5
      
        params['ord_threshold_estimate'] = ordinal_threshold_estimate
        num_list = list(range(1,num_trials,step))
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
    
if __name__ == '__main__':

    # save_folder = '/home/amyli/Documents/Results/2020_09_29_11_add_ord_pref_RD_100_saf_' + key +'/'
    save_folder = 'C:/Users/amykj/Documents/Results/2020_10_28_12_add_ord_pref_RD_500_saf_-0.45/' 
    prior_path = 'C:/Users/amykj/Documents/dim3_run_0_prior_cov_sig_1.0.mat'
    step_sz = 10
    eval_posterior(save_folder,state_dim =3,num_trials = 240,num_runs = range(20,50),feedback_type = ['ord','pref','coactive'],path = prior_path,sig = 1.0,step = step_sz,spec_trial_only = True)
    
