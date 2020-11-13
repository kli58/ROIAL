#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:28:44 2020

@author: amyli
"""
import numpy as np
import scipy.io as io
import os,random
from random import randrange
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from simulate_feedback import (get_preference, get_ordinal_feedback)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from math import sqrt
from plot_individual import norm_objective_data



def acc_pref_ord_label(objective_func,samp_list,ord_thresh = []):
    preference_label =[]
    OR_label = []
    
    for i in samp_list:
        if i > 0:
            preference_label.append(get_preference(i, i-1, objective_func)) 
        if len(ord_thresh):
            OR_label.append(get_ordinal_feedback(i,ord_thresh, objective_func)) 
            
    if len(OR_label):
        return preference_label,OR_label
    else:
        return preference_label

def evaluate_acc_labels(save_folder,state_dim,num_trials,num_runs,feedback,
            posterior = True, samp_num = 50,alpha = 0.1,fig_num = 2,line_style ='-',
            normal = False,use_sampled =False,color = '#377eb8'):
    fig = plt.figure(fig_num)
    if posterior:
        trial_nums = list(range(1,num_trials))
    else:
        trial_nums = [num_trials - 1]
    acc_score_or = np.zeros((len(num_runs),len(trial_nums)))
    acc_score_pref = np.zeros((len(num_runs),len(trial_nums)))
    for (run,r) in zip(num_runs,range(len(num_runs))):    
        filename = save_folder + 'dim' + str(state_dim) + '_run_' \
                        + str(run)  +'.mat'  
        #posterior_model = io.loadmat(filename)['posterior_model'][0]
        posterior_list = io.loadmat(filename)['posterior_mean']
        or_thresh_list = io.loadmat(filename)['ord_thresh_est']
        or_true = io.loadmat(filename)['ord_thresh_true'][0]
        objective_values = io.loadmat(filename)['norm_data'].flatten()
     
        if use_sampled:
            samp_list = io.loadmat(filename)['sampled_point_idx'].flatten()
            if 'add' in filename:
                samp_list = io.loadmat(filename)['sampled_flt_idx'].flatten()
        else: 
            samp_list = random.sample(range(len(objective_values)), samp_num)

        for i in trial_nums:
            posterior_mean = posterior_list[i]
            if normal:
                min_val = np.min(posterior_mean)
                shifted_mean = (posterior_mean - min_val)
                posterior_mean = shifted_mean/np.max(shifted_mean)
            if 'ord' in feedback:
                #print(i-1)
                if 'add' in filename:
                    or_est = or_thresh_list
                else:
                    or_est = or_thresh_list[i-1]
                or_thresh = [-np.inf]#[np.min(or_est)-5]
                or_thresh.extend(np.cumsum(or_est))
                or_thresh.append(np.inf)
                #print('or_thresh',or_est)
                if normal: 
                    or_thresh = (or_thresh - min_val)/np.max(shifted_mean)
                pref_label_true,ord_label_true = acc_pref_ord_label(objective_values, samp_list,or_true)
                pref_label_pred,ord_label_pred = acc_pref_ord_label(posterior_mean, samp_list,or_thresh)
                #ord_label_true = 
                #score_or = accuracy_score(ord_label_true, ord_label_pred)
                score_or = np.sum(np.abs(np.array(ord_label_true) - np.array(ord_label_pred)))/len(ord_label_true)
                acc_score_or[r,i-1]=score_or
                #print('or: ',score_or)
            else:
                pref_label_true = acc_pref_ord_label(objective_values, samp_list)
                pref_label_pred = acc_pref_ord_label(posterior_mean, samp_list)
            score_pref = accuracy_score(pref_label_true, pref_label_pred)
            acc_score_pref[r,i-1] = 1- score_pref
            #print('pred: ',score_pref)

    
    if 'ord' in feedback:
        #plt.scatter(trial_nums,acc_score_or)
        plt.subplot(121)
        pref_mean_err = np.mean(acc_score_pref, axis = 0)
        print(pref_mean_err.shape)
        pref_stdev = np.std(acc_score_pref, axis = 0)
        print(pref_stdev)
        plt.title('pref label')
        plt.ylim([0,0.6])
        plt.fill_between(np.arange(len(trial_nums)), pref_mean_err - pref_stdev, 
                             pref_mean_err + pref_stdev, alpha = alpha, color = color)
        plt.plot(np.arange(len(trial_nums)), pref_mean_err, color = color,linestyle= line_style)
        
        plt.subplot(122)
        ord_mean_err = np.mean(acc_score_or, axis = 0)
        #print(pref_mean_err)
        ord_stdev = np.std(acc_score_or, axis = 0)
        #print(pref_stdev)
        plt.fill_between(np.arange(len(trial_nums)), ord_mean_err - ord_stdev, 
                             ord_mean_err + ord_stdev, alpha = alpha, color = color)
        plt.plot(np.arange(len(trial_nums)), ord_mean_err, color = color,linestyle = line_style)
        #plt.legend(['pref','ord'])
        plt.title('or label')
        plt.ylim([0,2.0])


    else:
        pref_mean_err = np.mean(acc_score_pref, axis = 0)
        print(pref_mean_err.shape)
        pref_stdev = np.std(acc_score_pref, axis = 0)
        print(pref_stdev)
        
        plt.fill_between(np.arange(len(trial_nums)), pref_mean_err - pref_stdev, 
                             pref_mean_err + pref_stdev, alpha = alpha, color = color)
        plt.plot(np.arange(len(trial_nums)), pref_mean_err, color = color,linestyle= line_style)
    #    plt.scatter(trial_nums,acc_score_pref)
    #plt.suptitle(feedback)
        
    return pref_mean_err, pref_stdev,acc_score_pref
  
def evaluate_safe_performance(save_folder,state_dim,num_trials,num_runs,feedback, line_style = '-',
                          color = 'blue',alpha = 0.1,fig_num = 3):
    fig = plt.figure(fig_num)
    # Obtain the objective values over the runs.
    unsafe_list = np.zeros((len(num_runs),num_trials))
    unsafe_perc = np.zeros((len(num_runs),num_trials))
    for (run,r) in zip(num_runs,range(len(num_runs))):    
        filename = save_folder + 'dim' + str(state_dim) + '_run_' \
                        + str(run)  +'.mat'  
        if 'add' in filename:
            samp_list = io.loadmat(filename)['sampled_flt_idx'].flatten()[0:num_trials]
        else:
            samp_list = io.loadmat(filename)['sampled_point_idx'].flatten()[0:num_trials]
        objective_values = io.loadmat(filename)['norm_data'].flatten()
        objective_values = norm_objective_data(objective_values)
        #print(objective_values.shape)
        sampled_obj_val = objective_values[samp_list]
        saf_thresh = io.loadmat(filename)['params']['ord_b1']
        unsafe_idx_num = np.cumsum(sampled_obj_val < saf_thresh)
   
        unsafe_perc[r] = unsafe_idx_num

    
    unsafe_list = unsafe_perc
    unsafe_mean =  np.mean(unsafe_list,axis=0)
    unsafe_std = np.std(unsafe_list,axis =0)
    
    #plt.subplot(122)
    plt.plot(np.arange(num_trials),unsafe_mean,color=color,linestyle = line_style)
    plt.fill_between(np.arange(num_trials),unsafe_mean-unsafe_std,unsafe_mean+unsafe_std,alpha=alpha)
    plt.title('# of idx below b1')
    return 


if __name__ == '__main__':
    
    # Color-blind friendly palette: https://gist.github.com/thriveth/8560036
    CB_colors = ['#377eb8', '#ff7f00', '#4daf4a',  
                      '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00', '#f781bf']
     
  
    feedback = 'ord_pref_coactive_IG_saf_0.5_ord_0.225'
    save_folder = '/Users/amyli/Desktop/CalTech/gitrepo/Results/2020_09_17_12_' + feedback + '/'
    evaluate_acc_labels(save_folder,state_dim = 1,num_trials = 20,num_runs = range(1),feedback =['pref','ord'],
            posterior = True, samp_num = 50,alpha = 0.1,fig_num = 2,line_style ='-',
            normal = False,use_sampled =False,color = CB_colors[0])
   