#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:23:48 2020

@author: amyli
"""
import numpy as np
import scipy.io as io
import os
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_gp_2D(gx, gy, obj_func,mu, X_train_i,X_train_j, Y_train, title,save_folder = '',trial_num = '',save = False,b1_true = 0.33,b1_thresh = -0.5,plot_sample = True):
    
   
    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot(121)
        #ax = fig.gca(projection='3d')
    ct = ax.contour(gx, gy, obj_func,25,cmap = cm.coolwarm)
    ax.set_title('objective function')
    ax.clabel(ct, inline=True, fontsize=8)
    ax2 = plt.subplot(122)
    ct2 = ax2.contour(gx, gy, mu.reshape(gx.shape),25,cmap = cm.coolwarm)
    ax2.clabel(ct2, inline=True, fontsize=8)
    ax2.set_title('posterior mean ' + str(trial_num))
   
    if plot_sample:
        ax.scatter(X_train_j, X_train_i, s=2*np.arange(len(X_train_j)),cmap=cm.coolwarm,c=Y_train)
   
        ax2.scatter(X_train_j, X_train_i, s=2*np.arange(len(X_train_j)),cmap=cm.coolwarm, c=Y_train)
   
    plt.suptitle(title)
    if b1_thresh:
        if np.min(mu) < b1_thresh:
            level_alg = [np.min(mu),b1_thresh]
            ax2.contourf(gx, gy, mu.reshape(gx.shape), level_alg,colors = ['k','w'],alpha=0.2)
        level_true = [0,b1_true]
        ax.contourf(gx, gy, obj_func, level_true,colors = ['k','w'],alpha=0.2)
        
    if save:
        fig.savefig(save_folder + title + '_' + str(trial_num) + '.png')


def plot_1D_gp(mu, cov, X, X_train=None, Y_train=None,legend=True,true_function = None,title_1 = None,title_2 = None,lamb = 0,
               safe_thresh = 0, or_thresh= [], or_thresh_uncertain = None, or_true = [], save = False,or_len = 1,or_est_len = 0):
    fig = plt.figure()
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = np.sqrt(np.diag(cov))
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    if lamb:
        plt.fill_between(X, mu + lamb * uncertainty, mu - lamb * uncertainty, color = 'green', alpha=0.1)
    plt.plot(X, mu, label='posterior mean')
    color_list = ['#377eb8', '#ff7f00', '#4daf4a',  
                      '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00', '#f781bf']
    # for i, sample in enumerate(samples):
    #     plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        #colors = iter(cm.Greys()))
        # for (x,y) in zip(X_train,Y_train):
            # plt.plot(x,y,'rx',s=4*np.arange(len(X_train)))
        size = 20 * np.linspace(0.5, 1, len(X_train))
        plt.scatter(X_train,Y_train,marker = 'x',s=size,color = 'grey')
    if true_function is not None:
        plt.plot(X, true_function,label='objective function')
    if title_2:
        plt.title(title_2)
    if or_len == 1:
        plt.plot(X,or_true[1]*np.ones(X.shape),linestyle = 'dotted',color = 'grey')
    
    for i in range(or_est_len):
        plt.plot(X,or_thresh[i]*np.ones(X.shape),linestyle = '--',color = color_list[5],alpha = 0.4)
    plt.ylim([-0.5,1.5])
    plt.ylabel('f(x)')
    plt.xlabel('x')
    if legend:
        plt.legend()
    if save:
        fig.savefig(save_folder + title_1 + '.png')
    return save_folder + title_1 + '.png'
def norm_objective_data(data):
    shift_data = data - np.min(data)
    if np.max(shift_data):
        norm_data = shift_data / np.max(shift_data)
    else:
        norm_data = shift_data
    return norm_data

def plot_results(save_folder,state_dim,run_num,feedback,safe_thresh = 0,
                 b1 = False,posterior_change =False,save= False, or_length = 1,lamb = 0,
                 title = None, or_est_len = 0,subset = False):
    filename = save_folder + 'dim' + str(state_dim) + '_run_' \
                    + str(run_num)  +'.mat'  
    
    

    objective_values = io.loadmat(filename)['norm_data']
    objective_values = norm_objective_data(objective_values)
    
    or_true = [0,io.loadmat(filename)['sub_params']['ord_b1'][0]]
    ord_b1 = io.loadmat(filename)['sub_params']['ord_b1'][0]
    if subset:
        add_file = save_folder + 'dim' + str(state_dim) + '_run_' + str(run_num)  +'_post_process.mat'
        posterior_list =  io.loadmat(add_file)['posterior_mean']        
        sampled_idx = io.loadmat(filename)['sampled_flt_idx'].flatten()
        b1_alg_org = io.loadmat(filename)['ord_thresh_est'].flatten()[0]
    else:
        posterior_list = io.loadmat(filename)['posterior_mean']
        b1_alg_org = io.load(filename)['ord_thresh_est'].flatten()[0]
        sampled_idx = io.loadmat(filename)['sampled_point_idx'].flatten()
    #ord_b1 = np.percentile(objective_values,ord_b1 * 100)
    if state_dim == 1:
        posterior_cov_inverse = io.loadmat(filename)['posterior_cov_inverse']
        points_to_sample = io.loadmat(filename)['points_to_sample']
    #safe_thresh = io.loadmat(filename)['params']['ord_b1'][0]
        if b1:
            or_thresh_list = io.loadmat(filename)['b1_sampled'][0]
            #print(or_thresh_list)
        else:
            or_thresh_list = io.loadmat(filename)['ord_thresh_est']
    
    
    title_save = feedback +'_dim' + str(state_dim) + '_run_' + str(run_num)
    fig = plt.figure()
    if state_dim == 1:
        posterior_model = io.loadmat(filename)['posterior_model'][0]
        min_val = np.min(posterior_model['mean'][0])
        shifted_mean = (posterior_model['mean'][0] - min_val)
        mean = shifted_mean/np.max(shifted_mean)
        cov = np.linalg.inv(posterior_model['post_cov_inverse'][0])/ (np.max(shifted_mean)**2)
        ims = []
        if posterior_change:
            for i in range(len(posterior_list)):
                samp_idx = sampled_idx[:i+1]
                post_mean = posterior_list[i]
                post_cov_inv = posterior_cov_inverse[i]
                min_val = np.min(post_mean)
                shifted_mean = (post_mean - min_val)
                if np.max(shifted_mean):
                    mean = shifted_mean/np.max(shifted_mean)
                    cov = np.linalg.inv(post_cov_inv)/ (np.max(shifted_mean)**2)
                else:
                    mean = shifted_mean
                    cov = np.linalg.inv(post_cov_inv)
                if 'ord' in feedback:
                    or_thresh = or_thresh_list[-1]
                    or_thresh = np.cumsum(or_thresh)
                    #print(' ')
                    #print(or_thresh)
                    or_thresh = (or_thresh - min_val)/np.max(shifted_mean)
                    im = plot_1D_gp(mean, cov, points_to_sample, points_to_sample[samp_idx], objective_values.flatten()[samp_idx],
                               legend=True,true_function=objective_values.flatten(),title_1 = title_save + '_' + str(i), title_2 = title,
                               lamb = lamb,safe_thresh=safe_thresh,
                               or_thresh = or_thresh, or_true = or_true,save = save,or_len = or_length, or_est_len = or_est_len) 
                else:
                    im = plot_1D_gp(mean, cov, points_to_sample, points_to_sample[samp_idx], objective_values.flatten()[samp_idx],
                               legend=False,true_function=objective_values.flatten(),title=title + '_' +str(i),safe_thresh=safe_thresh,
                               save = save) 
                ims.append([im]) 

        else:
            samp_idx = sampled_idx
            im = plot_1D_gp(mean, cov, points_to_sample, points_to_sample[samp_idx], 
                       objective_values.flatten()[samp_idx],legend=True,true_function=objective_values.flatten(),title=title,safe_thresh=safe_thresh,save = save)  
    if state_dim ==2:

        num_pts = [30, 30]   # 30-by-30 grid
        x_vals = np.linspace(0, num_pts[0], num_pts[0])
        y_vals = np.linspace(0, num_pts[1], num_pts[1])
        Y, X = np.meshgrid(x_vals, y_vals)
         
   
        X_train = np.unravel_index(sampled_idx, num_pts)
        Y_train = objective_values.flatten()[sampled_idx]
        X_train_i = np.squeeze(X_train[0])
        X_train_j = np.squeeze(X_train[1])
       
    
        if posterior_change:
            for i in range(len(sampled_idx)):
                post_mean = posterior_list[i]
                min_val = np.min(post_mean)
                shifted_mean = (post_mean - min_val)
                mean = shifted_mean/np.max(shifted_mean)
                # mean = norm_objective_data(post_mean)
                b1_alg = (b1_alg_org - min_val)/np.max(shifted_mean)
                plot_gp_2D(Y,X, objective_values,mean, X_train_i[0:i],X_train_j[0:i], Y_train[0:i],title,save_folder,trial_num = i,save=save,b1_true = ord_b1,b1_thresh= b1_alg)          
        else:
            mean = norm_objective_data(posterior_list[-1])
            plot_gp_2D(Y,X, objective_values,mean, X_train_i,X_train_j, Y_train[0:i],title,save_folder,trial_num = i,save=save,b1_true= ord_b1,b1_thresh = b1_alg)  
            
    if state_dim == 3:
        num_pts = [20, 20,20]   # 30-by-30 grid
        x_vals = np.linspace(0, num_pts[0], num_pts[0])
        y_vals = np.linspace(0, num_pts[1], num_pts[1])
        Y, X = np.meshgrid(x_vals, y_vals)
         
        #fig = plt.figure(figsize = (7.2, 4.76))
        #ax = fig.gca(projection='3d')
        #surf = ax.plot_surface(Y, X, sample, cmap=cm.coolwarm, linewidth=0, alpha=0.2,antialiased=False)
        X_train_i =[]
        X_train_j = []
        Y_train = []
        for i in range(len(sampled_idx)):
            idx = np.unravel_index(sampled_idx[i],num_pts)
            X_train_i.append(idx[0])
            X_train_j.append(idx[1])
            Y_train.append(objective_values.flatten()[sampled_idx[i]])

        posterior_3d = posterior_list[-1]
        min_val = np.min(posterior_3d)
        shifted_mean = (posterior_3d - min_val)
        posterior_3d = shifted_mean/np.max(shifted_mean)
        posterior_3d = posterior_3d.reshape(num_pts)
        for j in range(num_pts[2]):
            mean = posterior_3d[:,:,j]
            plot_gp_2D(Y,X, objective_values[:,:,j],mean, X_train_i,X_train_j, Y_train,title + str(j),save_folder,trial_num = i,save=save,safe = ord_b1,plot_sample = False)  
                
        
        #mu = posterior_model['mean']     
        #plot_gp_2D(Y,X, objective_values,mean, X_train_i,X_train_j, Y_train,title,posterior_list,save_folder,posterior,save)          

def plot_add_GP_results(save_folder,state_dim,run_num,feedback,safe_thresh = 0,
                 b1 = False,posterior =False,save= False, or_length = 1,
                 title = None, or_est_len = 0,gif = False,post = False):
    filename = save_folder + 'dim' + str(state_dim) + '_run_' \
                    + str(run_num)  +'.mat'  
#    posterior_model = io.loadmat(filename)['posterior_model'][0]
    #points_to_sample = io.loadmat(filename)['points_to_sample']

    objective_values = io.loadmat(filename)['norm_data']
    objective_values = norm_objective_data(objective_values)
    if 'add' in filename:
        sampled_idx = io.loadmat(filename)['sampled_multi_idx']
        
    else:
        sampled_idx = io.loadmat(filename)['sampled_point_idx'].flatten()
    if post:
        add_file = save_folder + 'dim' + str(state_dim) + '_run_' + str(run_num)  +'_post_process.mat'
        posterior_list =  io.loadmat(add_file)['posterior_mean']
    else:
        posterior_list = io.loadmat(filename)['posterior_mean']
   
    

    if not title:
        title = feedback +'_dim' + str(state_dim) + '_run_' + str(run_num)
    fig = plt.figure()
   
      

    num_pts = [30, 30]   # 30-by-30 grid
    x_vals = np.linspace(0, num_pts[0], num_pts[0])
    y_vals = np.linspace(0, num_pts[1], num_pts[1])
    Y, X = np.meshgrid(x_vals, y_vals)
     
    #fig = plt.figure(figsize = (7.2, 4.76))
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(Y, X, sample, cmap=cm.coolwarm, linewidth=0, alpha=0.2,antialiased=False)
    X_train_i =[]
    X_train_j = []
    Y_train = []
    for i in range(len(sampled_idx)-1):
        if 'add' in filename:
            X_train_i.append(sampled_idx[i][0])
            X_train_j.append(sampled_idx[i][1])
            Y_train.append(objective_values[tuple(sampled_idx[i])])
        else:
            idx = sampled_idx[i]
            m = int(idx/num_pts[0])
            n = idx%num_pts[0]
            X_train_i.append(m)
            X_train_j.append(n)
            Y_train.append(objective_values.flatten()[sampled_idx[i]])
        if posterior:
            post_mean = posterior_list[i].flatten()
            
            min_val = np.min(post_mean)
            shifted_mean = (post_mean - min_val)
            mean = shifted_mean/np.max(shifted_mean)
            plot_gp_2D(Y,X, objective_values,mean, X_train_i,X_train_j, Y_train,title,save_folder,posterior,trial_num = i,save=save)          
        else:
            plot_gp_2D(Y,X, objective_values,mean, X_train_i,X_train_j, Y_train,title,save_folder,posterior,trial_num = i,save=save)  
            break
        #mu = posterior_model['mean']     
        #plot_gp_2D(Y,X, objective_values,mean, X_train_i,X_train_j, Y_train,title,posterior_list,save_folder,posterior,save)          

if __name__ == '__main__':
    
    save_folder = '/Users/amyli/Desktop/Results/ord_pref_RD_100/'
    plot_results(save_folder,state_dim = 2,run_num = 0, feedback = '',posterior_change = True, subset = True)
# for run_num in run_num:
#     plot_results(save_folder,state_dim,run_num, feedback,posterior = True, lamb = 0.0,title = ' slice ', save=True)
# plt.close('all')