#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:51:16 2020

@author: amyli
"""
import numpy as np
import scipy.io as io
import os,random,logging
from simulate_feedback import (get_preference,determine_ordinal_threshold, get_ordinal_feedback)
from gp_utility import information_gain_acquisition, feedback
from os.path import dirname
from datetime import datetime


def run_individual_GP(feedback_type,params,GP_prior_cov,GP_prior_cov_inv,data,f_map,logger):
    #initiate setting
    tr = params['curr_trials'] 
    ordinal_threshold_estimate = params['ord_threshold_estimate']
    ordinal_noise = params['ordinal_noise']
    saf_thresh = params['saf_thresh']
    preference_noise = params['pref_noise']
   
    #setup specfic feedback-related params
    num_category = params['num_category']
    ordinal_noise = params['ordinal_noise']
    
    #parse dataset
    pref_data_idxs = data['pref_data_idxs']
    pref_labels = data['pref_labels']

    OR_point_idx = data['ord_data_idxs']
    OR_labels = data['ord_labels']
    

    # Update the Gaussian process based on preference/ordinal feedback:
    posterior_model = feedback(pref_data_idxs, pref_labels, [], [], GP_prior_cov_inv,
                            OR_point_idx,ordinal_threshold_estimate,OR_labels, ordinal_noise,
                            preference_noise, [],[],r_init = f_map, logger=logger)    

    # Sample new points at which to query for a preference:

    if params['query_type'] == 'IG' and tr > 0:
        sampled_point_idx = information_gain_acquisition(OR_point_idx[tr-1],OR_point_idx,OR_labels, GP_prior_cov_inv,
                        posterior_model,params['IG_it'],num_category,ordinal_threshold_estimate,
                        ordinal_noise,preference_noise,feedback_type,saf_thresh,logger,method = params['saf_method'])
        
    elif params['query_type'] == 'post_process':
        sampled_point_idx = []
   
    else:
        sampled_point_idx = [random.randint(0,GP_prior_cov.shape[0]-1)]
    
    return sampled_point_idx, posterior_model['mean']


def init_points_to_sample(num_pts,state_dim):
    if state_dim == 2:
        points_to_sample = np.empty((num_pts[0] * num_pts[1], state_dim))
        x_vals = np.linspace(0, 1, num_pts[0])
        y_vals = np.linspace(0, 1, num_pts[1])
        xv,yv = np.meshgrid(x_vals,y_vals,indexing = 'ij')
        points_to_sample = np.array([xv.flatten(),yv.flatten()]).T
             
    elif state_dim == 3:
        x_vals = np.linspace(0, 1, num_pts[0])
        y_vals = np.linspace(0, 1, num_pts[1])
        z_vals = np.linspace(0, 1, num_pts[2])
        xv,yv,zv = np.meshgrid(x_vals,y_vals,z_vals,indexing = 'ij')
        points_to_sample = np.array([xv.flatten(),yv.flatten(),zv.flatten()]).T

    else:
        x_vals = np.arange(0, 1, 1.0/num_pts.item()).reshape(-1, 1)
        points_to_sample = x_vals

    return points_to_sample
    
def tdot(mat):
    return np.dot(mat, mat.T)

def kernel(X,variance, lengthscales):
    X = X/lengthscales
    Xsq = np.sum(np.square(X),1)
    r2 = -2. * tdot(X) + (Xsq[:, None] + Xsq[None, :])
    #util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
    np.fill_diagonal(r2, 0.)
    r_sq = np.clip(r2, 0, np.inf)
    # RBF
    cov = variance * np.exp(-0.5 * r_sq)
    return cov 

def gp_kernel(points_to_sample,signal_variance,lengthscales, GP_noise_var):
    
    # Points over which objective function was sampled. These are the points over
    # which we will draw samples.  

    num_pts_sample, state_dim = points_to_sample.shape
    GP_prior_cov = kernel(points_to_sample,signal_variance,lengthscales)

    GP_prior_cov += GP_noise_var * np.eye(num_pts_sample)
    GP_prior_cov_inv = np.linalg.inv(GP_prior_cov) 
    return GP_prior_cov, GP_prior_cov_inv

def run_GP(filename,feedback_type,sub_params,save_folder,run_num,log_file):
       
    # create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    
    # create a logging format
    logger = logging.getLogger('sim_log')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('sub params: ' + str(sub_params))
    
    iterations =sub_params['num_trials']
    D = sub_params['D']
    #load objective function
    objective = sub_params['objective_function']
    grid_shape = np.array(objective.shape)
    ordinal_threshold = sub_params['ordinal_threshold_true']
    ordinal_threshold_estimate = sub_params['model_params']['ord_threshold_estimate']
    num_action = np.prod(grid_shape)
  
    print(num_action)

    points_to_sample = init_points_to_sample(grid_shape, D)
    actions = list(range(num_action))
    
      
    ord_labels = []
    pref_labels = []
    coactive_label = []
    sampled_idx = {}
    sampled_multi_idx = []
    pref_idxs = []
    ord_idxs = []
    coactive_idxs = []
    posterior_mean_list = []
    decomp_list = []
    sampled_flt_idx = []
    sampled_coord_points = []
    posterior_mean_whole = np.zeros(grid_shape)
    relab_sampled_idx = []
    pref_idxs_post = []
    coactive_idxs_post = []
    sampled_virtual_idx = []
    cts = 0
    
    # return a randon idx
    joint_idx =  tuple([random.randint(0,grid_shape[i]-1) for i in range(D)])
    
    for t in range(1,iterations):
        #run evidence maximization every N_cyc iterations
        print('iteration: ',t)
        
        flt_idx = np.ravel_multi_index(joint_idx, grid_shape)
        print(flt_idx)
        if flt_idx not in sampled_idx.values():
            pts = cts
            cts += 1
            sampled_idx[pts] =flt_idx
            sampled_coord_points.append(points_to_sample[flt_idx])
        else:
            print('flt_idx: ',flt_idx)
            print('sampled_idx: ',sampled_idx)
            pts = list(sampled_idx.keys())[list(sampled_idx.values()).index(flt_idx)]
            print('pts: ',pts)
                
        if 'ord' in feedback_type:
            
            ord_label = get_ordinal_feedback(tuple(joint_idx), ordinal_threshold, objective,add_GP = True,noise = sub_params['model_params']['ord_nos'])

        if 'pref' in feedback_type and len(sampled_flt_idx) > 0:
            preference = get_preference(tuple(joint_idx),tuple(sampled_multi_idx[-1]), objective,add_GP = True,noise = sub_params['model_params']['pref_nos'])
            #pref_idxs.append([joint_idx, sampled_idx[-1]])
            
            pref_idxs.append([pts, relab_sampled_idx[-1]]) #TODO: assume no action will be sampled again
            pref_label = [1 - preference, preference]
            pref_labels.append(pref_label)
            
            pref_idxs_post.append([flt_idx,sampled_flt_idx[-1]])
        
         #TODO check which one should be deleted sampled_flt_idx or sampled_idx (which is a dictionary)
        sampled_flt_idx.append(flt_idx) #actual flatten index corresponding to action space
        relab_sampled_idx.append(pts) # index determined by the order of sampling 
        sampled_multi_idx.append(tuple(joint_idx))
        ord_idxs.append(pts)
        ord_labels.append(ord_label)
        
       
        sub_idx_rand = [] 
        
        if sub_params['dropout_method'] == 'RD':
            print('start rd')
            actions_to_sample = list(set(actions) - set(sampled_flt_idx) - set(sampled_virtual_idx))
            if sub_params['rd_sz'] == num_action:
                sub_idx_rand = actions
            else:
                sub_idx_rand = random.sample(actions_to_sample, sub_params['rd_sz'])
        
        sub_whole_idx = np.append(list(sampled_idx.values()),sub_idx_rand)

        query_pts = points_to_sample[sub_whole_idx,:]

        sub_data_points = np.array(query_pts)
        

        print(sub_data_points.shape)
        #re-evaluate sub covariance matrix
       
        GP_prior_cov, GP_prior_cov_inv = gp_kernel(sub_data_points,sub_params['model_params']['signal_variance'],sub_params['model_params']['lengthscales'], sub_params['model_params']['GP_noise_var'])
        
        #re-gernerate idx and corresponding data according to the subspace
        data = {}
        
        if len(pref_labels):
            data['pref_labels'] = np.array(pref_labels)[:,1]
            data['pref_data_idxs']= np.array(pref_idxs)
        else:
            data['pref_labels'] = np.array([])
            data['pref_data_idxs']= np.zeros((0,2))
        data['ord_data_idxs'] = np.array(ord_idxs)
        data['ord_labels'] = np.array(ord_labels)
        
        #pass these data points to a GP
        params = sub_params['model_params']
        params['curr_trials'] = len(sampled_flt_idx)
        sampled_point_idx, posterior_mean = run_individual_GP(feedback_type,params,GP_prior_cov,GP_prior_cov_inv,data,[],logger)
       
        
        sampled_action = np.array(np.unravel_index(sub_whole_idx[sampled_point_idx[0]],grid_shape))

        print('sampled action: ', sampled_action)
        joint_idx = sampled_action
        #update subplane posterior
        sub_whole_idx = np.unravel_index(sub_whole_idx, grid_shape)
        posterior_mean_whole[sub_whole_idx] = posterior_mean
        
        #update subplane posterior
        sub_whole_idx = np.unravel_index(sub_whole_idx, grid_shape)
        posterior_mean_whole[sub_whole_idx] = posterior_mean
        posterior_mean_list.append(posterior_mean_whole.copy())

    io.savemat(save_folder + 'dim' + str(D) + '_run_' + str(run_num)  +'.mat', { 'pref_labels': pref_labels,'ord_labels':ord_labels, 'sub_params':sub_params,
            'posterior_mean':posterior_mean_list,'decomp':decomp_list,'ord_thresh_true':ordinal_threshold,'ord_thresh_est':ordinal_threshold_estimate,
            'sampled_point_idx':sampled_idx,'sampled_multi_idx':sampled_multi_idx,'sampled_flt_idx':sampled_flt_idx,'pref_idx_post':pref_idxs_post,
            'norm_data':objective})
    
        
    return

def run_simulation(root_directory,feedback_type,sub_params,save_folder = ''):
    if not save_folder:
        dat = datetime.now()
        dt_string = dat.strftime("%Y_%m_%d_%H")
        key = '_' + str(sub_params['rd_sz'])
        if sub_params['model_params']['ord_nos']:
            key += '_noise_' + str(sub_params['model_params']['ord_nos'])
        save_folder = root_directory + '/Results/' + dt_string + '_sub_' + '_'.join(feedback_type) + '_' +sub_params['dropout_method'] + key +'_saf_' + str(sub_params['model_params']['saf_thresh'])+'/'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder) 

    D = sub_params['D']
    log_file = save_folder + 'dim_' + str(D) + '.log'
    
    file_part = sub_params['file_part']

    for i in sub_params['run_nums']:
        #load objective functions
        if 'hart' in file_part:
            filename = file_part
        else:
            filename = file_part + str(i) + '.mat'
        data = io.loadmat(filename)
        objective_function = data['sample']
        
        #update data struct
        sub_params['objective_function'] =objective_function
        ordinal_threshold_true = determine_ordinal_threshold(sub_params['num_category'],sub_params['ord_b1'],sub_params['ord_delta'],sub_params['objective_function'], sub_params['ord_perct'])
        sub_params['ordinal_threshold_true'] =ordinal_threshold_true
        print('ordinal threshold true: ', ordinal_threshold_true)   
        run_GP(filename,feedback_type, sub_params, save_folder,i,log_file)
    
    
    
file_dir = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__':
    root_directory = dirname(dirname(dirname(file_dir))) 
    feedback_type = ['ord','pref']
    model_params = {'ord_nos':0,'pref_nos':0.06,'signal_variance': 1.0, 'lengthscales': [0.15, 0.15,0.15],'GP_noise_var': 0.01,
                 'IG_it':1000,'query_type':'IG','saf_thresh': 0.0, 'saf_method': 'none',
                 'pref_noise':0.015,'num_category':5, 'ordinal_noise':0.1}
    
    delta_int = 2/model_params['num_category']
    ordinal_threshold_estimate = np.array([delta_int]*(model_params['num_category']-1))
    ordinal_threshold_estimate[0] = -0.5
    model_params['ord_threshold_estimate'] = ordinal_threshold_estimate
    model_params['saf_method'] = 'IG_ucb'
    model_params['saf_thresh'] = -0.45
    sub_params = {'model_params':model_params,'dropout_method':'RD','rd_sz':500,'run_nums': range(1),'num_trials':10,'D':3,'num_category':5,'ord_perct':False,
                   'ord_b1':0.33,'ord_delta':[0.2,0.15,0.13],'file_part': file_dir + '/Sampled_functions_2D/30_by_30/Sampled_objective_'}
    sub_params['file_part'] = file_dir + '/Sampled_functions_3D/Sampled_objective_' 
    run_simulation(root_directory,feedback_type,sub_params)
    

