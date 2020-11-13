#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 13:59:02 2020

@author: amyli
"""

"""
These are functions used for learning a Bayesian utility model given preference 
data and directional human-guided feedback, and for drawing samples from this 
model.

The sigmoidal link function is used to capture noise in the user's preferences.
"""
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
import sys,itertools
import random
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
#sampled_idxs: have the same format as preference, current and previous samples
# posterior model needs to contain current sample nnd previous sample

def init_points_to_sample(grid_shape,discretization,lbs,ubs,state_dim):
    pts_dict = {}
    cts = 0
    if state_dim == 2:
        points_to_sample = np.empty((np.prod(grid_shape), state_dim))
        x_vals = np.linspace(lbs[0], ubs[0], grid_shape[0])
        y_vals = np.linspace(lbs[1], ubs[1], grid_shape[1])
        for i, x_val in enumerate(x_vals):
            for j, y_val in enumerate(y_vals):
                idx = np.ravel_multi_index([i,j], grid_shape)
                points_to_sample[idx, :] = [x_val, y_val]
                 
    elif state_dim == 3:
        points_to_sample = np.empty((np.prod(grid_shape), state_dim))
        x_vals = np.linspace(lbs[0], ubs[0], grid_shape[0])
        y_vals = np.linspace(lbs[1], ubs[1], grid_shape[1])
        z_vals = np.linspace(lbs[2], ubs[2], grid_shape[2])
        #xv, yv,zv = np.meshgrid(x, y,z)
        # Define grid of points over which to evaluate the objective function:
        #num_sample_points = np.prod(num_pts)  # Total number of points in grid
        points_to_sample = np.empty((np.prod(grid_shape), state_dim))
        # Put the points into list format:
        for i, x_val in enumerate(x_vals):
            for j, y_val in enumerate(y_vals):
                for k, z_val in enumerate(z_vals):
                    idx = np.ravel_multi_index([i,j,k], grid_shape)
                    points_to_sample[idx, :] = [x_val, y_val,z_val]
   
    elif state_dim == 4:
        points_to_sample = np.empty((np.prod(grid_shape), state_dim))
        x_vals = np.linspace(lbs[0], ubs[0], grid_shape[0])
        y_vals = np.linspace(lbs[1], ubs[1], grid_shape[1])
        z_vals = np.linspace(lbs[2], ubs[2], grid_shape[2])
        u_vals = np.linspace(lbs[3], ubs[3], grid_shape[3])
        xv,yv,zv,uv = np.meshgrid(x_vals,y_vals,z_vals,u_vals,indexing = 'ij')
        points_to_sample = np.array([xv.flatten(),yv.flatten(),zv.flatten(),uv.flatten()]).T
        #xv, yv,zv = np.meshgrid(x, y,z)
        # Define grid of points over which to evaluate the objective function:
        #num_sample_points = np.prod(num_pts)  # Total number of points in grid
        # points_to_sample = np.empty((np.prod(grid_shape), state_dim))
        # Put the points into list format:
        # for i, x_val in enumerate(x_vals):
        #     for j, y_val in enumerate(y_vals):
        #         for k, z_val in enumerate(z_vals):
        #             for u, u_val in enumerate(u_vals):
        #                 idx = np.ravel_multi_index([i,j,k,u], grid_shape)
        #                 points_to_sample[idx, :] = [x_val, y_val,z_val,u_val]
   
    
    return points_to_sample

def run_individual_GP(feedback_type,params,GP_prior_cov,GP_prior_cov_inv,data,f_map,logger, post_process = False):
    #initiate setting
    tr = params['curr_trials']
    cov_scale = params['cov_scale']
    coeff = params['coeff']
    ordinal_threshold_estimate = params['ord_threshold_estimate']
    ordinal_noise = params['ordinal_noise']
    saf_thresh = params['saf_thresh']
    preference_noise = params['pref_noise']

    #setup specfic feedback-related params
    #ord params
    num_category = params['num_category']
    ordinal_noise = params['ordinal_noise']
    
    
    #parse dataset
    pref_data_idxs = data['pref_data_idxs']
    pref_labels = data['pref_labels']

    OR_point_idx = data['ord_data_idxs']
    OR_labels = data['ord_labels']
    
    coactive_idx = data['coactive_idxs']
    coactive_labels = data['coactive_label']
   

    # Update the Gaussian process based on preference/ordinal feedback:
    posterior_model = feedback(pref_data_idxs, pref_labels, [], [], GP_prior_cov_inv,
                            OR_point_idx,ordinal_threshold_estimate,OR_labels, ordinal_noise,
                            preference_noise, coactive_idx,coactive_labels, cov_scale,coeff,r_init = f_map, logger=logger)    

    if not post_process:
        sampled_point_idx,_ = information_gain_acquisition(OR_point_idx[tr-1],OR_point_idx,OR_labels, GP_prior_cov_inv,
                            posterior_model,params['IG_it'],num_category,ordinal_threshold_estimate,
                            ordinal_noise,preference_noise,feedback_type,saf_thresh,logger,method = params['saf_method'])
       
    else:
         sampled_point_idx = ''   
    return sampled_point_idx, posterior_model['mean']


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

def gp_prior(points_to_sample,signal_variance,lengthscales, GP_noise_var):
    
    # Points over which objective function was sampled. These are the points over
    # which we will draw samples.  
    #points_to_sample = init_points_to_sample(num_pts, state_dim)
    #num_pts_sample = np.prod(points_to_sample.shape)
    num_pts_sample, state_dim = points_to_sample.shape
    GP_prior_cov = kernel(points_to_sample,signal_variance,lengthscales)
    GP_prior_cov += GP_noise_var * np.eye(num_pts_sample)
    GP_prior_cov_inv = np.linalg.inv(GP_prior_cov) 
    return GP_prior_cov, GP_prior_cov_inv

def information_gain_acquisition(sampled_idxs,OR_point_idxs,OR_labels, 
        GP_prior_cov_inv,posterior_model,M,ord_cat,ord_thresh,ord_noise,
        pref_noise,feedback_type,saf_thresh,logger,evd = False,method = 'none'
        ,cov_scale = 1):
    # Unpack model posterior:
    mean = posterior_model['mean']
    cov_evecs = posterior_model['cov_evecs']
    cov_evals = posterior_model['cov_evals']
    cov = np.linalg.inv(posterior_model['post_cov_inverse'])
    sigma = np.sqrt(np.diag(cov))
    num_features = len(mean)
    #ord_noise = ord_noise * np.max(np.abs(ord_thresh))
     # Obtain M sampled reward functions
    first_part = cov_scale * cov_evecs @ np.diag(np.sqrt(cov_evals))
    # Draw the samples:
    X = np.random.normal(size = num_features*M)
    X = np.reshape(X,(M,num_features))
    #mean_reshape = np.reshape(np.array(list(mean)*M),(M,num_features))
    #R = np.real(mean_reshape +  X @ first_part )
    
    R = np.zeros((M,num_features))
    for i in range(M):
    # Sample reward function from GP model posterior:
        R_i = mean + first_part @ X[i]
        R[i] = np.real(R_i)      # Store sampled reward function
       
    #R = R.reshape(R,(M,num_features)   
    safe_idx = range(num_features)
    IG = []
    
    b1_samples = 0

    if method == 'IG_ucb' and saf_thresh: #TODO: change this to safe constrain
            print('evaluate saf set')
            ucb = mean + saf_thresh * sigma
            safe_idx = [p for p in range(num_features) if ucb[p] > ord_thresh[0]]
    
    print('safety set size', len(safe_idx))
    if len(safe_idx) == 0:
        safe_idx = range(num_features)   
        
    for m in safe_idx:
        # loop through all available (safe) points to sample
        p = np.zeros((M,2,ord_cat))
        y1 = R[:,m] # new points 
        y2 = R[:,int(sampled_idxs)] #already sampled
        
        #all possible combination of labels
        for (s,o) in list(itertools.product([0,1], range(ord_cat))):
            if 'pref' in feedback_type and 'ord' in feedback_type:
                p[:,s,o] = ord_likelihood(y1,ord_thresh,o+1,ord_noise) * pref_likelihood([y1,y2],s,pref_noise)
            if 'pref' in feedback_type and not 'ord' in feedback_type:
                p[:,s,o] = pref_likelihood([y1,y2],s,pref_noise)
            if 'ord' in feedback_type and not 'pref' in feedback_type:
                p[:,s,o] = ord_likelihood(y1,ord_thresh,o+1,ord_noise)
            #if np.min(p[:,s,o]) == 0: print(s,o)

        h = - np.sum(np.sum(p * np.log2(p),axis=2),axis=1)
        p_avg = np.mean(p,axis = 0)
        H1 = - np.sum(p_avg * np.log2(p_avg))
        H2 = 1/M * np.sum(h)
        if np.isnan(H1) or np.isnan(H2): 
            logger.info(str(H1) + str(H2))
        IG.append(H1-H2)
    ucb = mean + sigma


    try:
        return [safe_idx[np.nanargmax(IG)]],np.mean(b1_samples)
    except:
        return [safe_idx[random.randint(0,len(safe_idx)-1)]],np.mean(b1_samples)
    



def feedback(pref_data, pref_labels, HG_points, HG_regions, GP_prior_cov_inv,
             OR_points, OR_threshold_estimate, OR_labels, ordinal_noise,
             preference_noise,coact_idx,coact_labels, cov_scale = 1, coeff = {'pref':1,'ord':1,'coact':1,'direc':1}, r_init = [], logger = None,min_ord_thresh = -1):
    """
    Function for updating the GP preference model given data.
    
    Inputs (m = number of preferences, n = pieces of human-guided feedback):
        1) pref_data: m x 2 NumPy array in which each row contains the indices
           of the two points compared in a specific preference; expected to be
           of integer type.
        2) pref_labels: length-m NumPy array, in which each element is 0 (1st 
           queried point preferred), 1 (2nd point preferred), or 0.5 (no 
           preference).
        3) HG_points: length-n NumPy array, in which each element records the 
           index of a point at which human-guided feedback was given.
        4) HG_regions: length-n list, in which the i-th element is a NumPy array
           with the indices of points in which the utility function is expected
           to improve due to the i-th piece of human-guided feedback.
        5) GP_prior_cov_inv: d-by-d NumPy array, where d is the number of 
           points over which the posterior is to be sampled 
        6) preference_noise: positive scalar parameter. Higher values indicate
           larger amounts of noise in the expert preferences.
        7) (Optional) cov_scale: parameter between 0 and 1; this is multiplied 
           to the posterior standard deviation when sampling in the advance 
           function, so values closer to zero result in less exploration.
        8) (Optional) initial guess for convex optimization; length-d NumPy
           array when specified.
               
    Output: the updated model posterior, represented as a dictionary of the 
           form {'mean': post_mean, 'cov_evecs': evecs, 'cov_evals': evals};
           post_mean is the posterior mean, a length-d NumPy array in which d
           is the number of points over which the posterior is to be sampled.
           cov_evecs is an d-by-d NumPy array in which each column is an
           eigenvector of the posterior covariance, and evals is a length-d 
           array of the eigenvalues of the posterior covariance.
    """   
    num_features = GP_prior_cov_inv.shape[0]
    
    #OR_regions modification
    # if min_ord_thresh >= OR_threshold_estimate[0]:
    #     min_ord_thresh = OR_threshold_estimate[0] - 0.1
    # OR_regions = [min_ord_thresh]
    
    # OR_regions.extend(np.cumsum(OR_threshold_estimate))
    OR_regions = [-5] #[-np.inf]
    OR_regions.extend(np.cumsum(OR_threshold_estimate))
    OR_regions = np.append(OR_regions,5)


    #OR_regions = [OR_threshold_estimate[0]+ min_ord_thresh]
    #OR_regions.extend(np.cumsum(OR_threshold_estimate))
    
    #OR_regions = OR_threshold_estimate
    
    # Remove any preference data recording no preference:
    pref_labels = pref_labels.flatten()
    pref_indices = np.where((pref_data[:, 0] != pref_data[:, 1]) & (pref_labels != 0.5))[0]
    pref_data = pref_data[pref_indices, :]
    pref_labels = pref_labels[pref_indices].astype(int)

    # Solve convex optimization problem to obtain the posterior mean reward 
    # vector via Laplace approximation:    
    if r_init == []:
        r_init = np.zeros(num_features)    # Initial guess
    # l = len(OR_threshold_estimate)
    # ordinal_noise_est = ordinal_noise * max(OR_regions[1]-OR_regions[0],np.max(np.abs(OR_threshold_estimate[1:l+1])))
    # if ordinal_noise_est > 0:
    #     ordinal_noise = ordinal_noise_est
    # if ordinal_noise <= 0.1:
    #     ordinal_noise = 0.1
    #elif ordinal_noise_est > 0 and ordinal_noise_est < 0.05:
    #    ordinal_noise = 4 * ordinal_noise_est
    #rint('or_noise: ',ordinal_noise)
    res = minimize(preference_GP_objective, r_init, args = (pref_data, pref_labels, 
                           HG_points, HG_regions, OR_points, OR_regions, OR_labels,coact_idx,coact_labels,
                           GP_prior_cov_inv, preference_noise, ordinal_noise,coeff), 
                   method = 'L-BFGS-B', jac = preference_GP_gradient)
    
    # The posterior mean is the solution to the optimization problem:
    post_mean = res.x

    if cov_scale > 0: # Calculate eigenvectors/eigenvalues of covariance matrix

        # Obtain inverse of posterior covariance approximation by evaluating the
        # objective function's Hessian at the posterior mean estimate:
        post_cov_inverse = preference_GP_hessian(post_mean,pref_data, pref_labels, 
                           HG_points, HG_regions, OR_points, OR_regions, OR_labels,coact_idx,coact_labels,
                           GP_prior_cov_inv, preference_noise, ordinal_noise,coeff,logger)
    
        # Calculate the eigenvectors and eigenvalues of the inverse posterior 
        # covariance matrix:
        evals, evecs = np.linalg.eigh(post_cov_inverse)
    
        # Invert the eigenvalues to get the eigenvalues corresponding to the 
        # covariance matrix:
        evals = 1 / evals
        
    else:   # cov_scale = 0; evecs/evals not used in advance function.
    
        evecs = np.eye(num_features)
        evals = np.zeros(num_features)
    
    # Return the model posterior:
    return {'mean': post_mean, 'cov_evecs': evecs, 'cov_evals': evals,'post_cov_inverse': post_cov_inverse,'or_noise':ordinal_noise}


#OR_regions--> b; OR_labels --> y    
def preference_GP_objective(f, pref_data, pref_labels, HG_points, HG_regions,
                            OR_points, OR_regions, OR_labels,coact_idx,coact_labels,
                            GP_prior_cov_inv, preference_noise, ordinal_noise,coeff):
    """
    Evaluate the optimization objective function for finding the posterior 
    mean of the GP preference model (at a given point); the posterior mean is 
    the minimum of this (convex) objective function.
    
    Inputs:
        1) f: the "point" at which to evaluate the objective function. This is
           a length-n vector, where n is the number of points over which the 
           posterior is to be sampled.
        2)-7): same as the descriptions in the feedback function. 
        
    Output: the objective function evaluated at the given point (f).
    """

    obj = 0.5 * f @ GP_prior_cov_inv @ f   # Initialize to term from prior
    
    # Process preference feedback:
    num_samples = pref_data.shape[0]
    
#    s_pos = np.zeros(num_samples)
#    s_neg = np.zeros(num_samples)
    for i in range(num_samples):   # Go through each pair of data points
        
        #s_pos[i] = pref_data[i,pref_labels[i]]
        #s_neg[i] = pref_data[i,1 - pref_labels[i]]
        
        data_pts = pref_data[i, :].astype('int')      # Data points queried in this sample
        label = pref_labels[i]
        
        # s_pos[i] = data_pts[label]
        # s_neg[i] = data_pts[1 - label]
        z = (f[data_pts[label]] - f[data_pts[1-label]]) / preference_noise
        obj -= coeff['pref'] *np.log(sigmoid(z))

    # if num_samples:
        
    #     s_pos = pref_data[np.arange(num_samples),pref_labels]
    #     s_neg = pref_data[np.arange(num_samples),1-pref_labels]
    #     z = (f[s_pos] - f[s_neg]) / preference_noise
    #z = (f[data_pts[label]] - f[data_pts[1-label]]) / preference_noise
    #obj -= coeff['pref'] *np.log(sigmoid(z))
        
        obj -= np.sum(coeff['pref'] *np.log(sigmoid(z)))
        #print('processed pref')
        
    for i in range(len(coact_idx)):   # Go through each pair of data points
        
        data_pts = coact_idx[i]    # Data points queried in this sample
        label = coact_labels[i]
        z = (f[data_pts[label]] - f[data_pts[1 - label]]) / preference_noise
        obj -= coeff['coact'] *np.log(sigmoid(z))
    # Process human-guided feedback:
    for i in range(len(HG_points)):
        
        z = (np.max(f[HG_regions[i]]) - f[HG_points[i]]) / preference_noise
        obj -= coeff['direc']*np.log(sigmoid(z))
        
    #process ordinal label:
    # for i in range(len(OR_points)):
    #     z1 = (OR_regions[OR_labels[i]] -  f[OR_points[i]])/ ordinal_noise
    #     z2= (OR_regions[OR_labels[i]-1] -  f[OR_points[i]])/ ordinal_noise
    #     obj -= np.log(norm.cdf(z1) - norm.cdf(z2))
        
    if len(OR_points):
        OR_regions = np.array(OR_regions)
        z1 = (OR_regions[OR_labels] -  f[OR_points])/ ordinal_noise
        z2 = (OR_regions[np.subtract(OR_labels,1)] -  f[OR_points])/ ordinal_noise
        obj -= coeff['ord']*np.sum(np.log(sigmoid(z1) - sigmoid(z2)))
    return obj



def preference_GP_gradient(f, pref_data, pref_labels, HG_points, HG_regions, 
                           OR_points, OR_regions, OR_labels,coact_idx,coact_labels,
                           GP_prior_cov_inv, preference_noise, ordinal_noise,coeff):
    """
    Evaluate the gradient of the optimization objective function for finding 
    the posterior mean of the GP preference model (at a given point).
    
    Inputs:
        1) f: the "point" at which to evaluate the gradient. This is a length-n
           vector, where n is the number of points over which the posterior 
           is to be sampled.
        2)-7): same as the descriptions in the feedback function. 
        
    Output: the objective function's gradient evaluated at the given point (f).
    """
    
    grad = GP_prior_cov_inv @ f    # Initialize to 1st term of gradient
    
    # Process preference feedback:
    num_samples = pref_data.shape[0]
    
    for i in range(num_samples):   # Go through each pair of data points
        
        data_pts = pref_data[i, :]     # Data points queried in this sample
        label = pref_labels[i]
        
        s_pos = int(data_pts[label])
        s_neg = int(data_pts[1 - label])
        
        z = (f[s_pos] - f[s_neg]) / preference_noise
        
        value = coeff['pref']*(sigmoid_der(z) / sigmoid(z)) / preference_noise
        
        grad[s_pos] -= value
        grad[s_neg] += value
    
    
#    if num_samples:
#        s_pos = pref_data[np.arange(num_samples),pref_labels]
#        s_neg = pref_data[np.arange(num_samples),1-pref_labels]
#        z = (f[s_pos] - f[s_neg]) / preference_noise
#        value = coeff['pref']*(sigmoid_der(z) / sigmoid(z)) / preference_noise
#        grad[s_pos] -= value
#        grad[s_neg] += value
        
    for i in range(len(coact_idx)):   # Go through each pair of data points
        
        data_pts = coact_idx[i]      # Data points queried in this sample
        label = coact_labels[i]
        
        s_pos = data_pts[label]
        s_neg = data_pts[1 - label]
        
        z = (f[s_pos] - f[s_neg]) / preference_noise
        
        value = coeff['coact']*(sigmoid_der(z) / sigmoid(z)) / preference_noise
        
        grad[s_pos] -= value
        grad[s_neg] += value
        
    # Process human-guided feedback:
    for i in range(len(HG_points)):
        
        HG_region = HG_regions[i]
        region_vals = f[HG_region]
        region_max = np.max(region_vals)
        
        s_neg = HG_points[i]  
        
        # Calculate value to use for updating the gradient
        z = (region_max - f[s_neg]) / preference_noise
        value = coeff['direc']*(sigmoid_der(z) / sigmoid(z)) / preference_noise
        
        grad[s_neg] += value
        
        # Consider all points in the HG feedback region at which f is maximized
        region_argmax_idxs = HG_region[np.where(region_vals == region_max)[0]]
        grad[region_argmax_idxs] -= value            
    
    if len(OR_points):
        OR_regions = np.array(OR_regions)
        z1 = (OR_regions[OR_labels] -  f[OR_points])/ ordinal_noise
        z2 = (OR_regions[np.subtract(OR_labels,1)] -  f[OR_points])/ ordinal_noise
        grad_i_terms = coeff['ord'] * 1/ordinal_noise * (sigmoid_der(z1) - sigmoid_der(z2)) / (sigmoid(z1) - sigmoid(z2))
        for i in range(len(OR_points)):
            grad[OR_points[i]] += grad_i_terms[i]
    return grad

def preference_GP_hessian(f, pref_data, pref_labels, HG_points, HG_regions, 
                           OR_points, OR_regions, OR_labels,coact_idx,coact_labels,
                           GP_prior_cov_inv, preference_noise, ordinal_noise,coeff,logger):
    """
    Evaluate the Hessian matrix of the optimization objective function for 
    finding the posterior mean of the GP preference model (at a given point).
    
    Inputs:
        1) f: the "point" at which to evaluate the Hessian. This is
           a length-n vector, where n is the number of points over which the 
           posterior is to be sampled.
        2)-7): same as the descriptions in the feedback function. 
        
    Output: the objective function's Hessian matrix evaluated at the given 
            point (f).
    """
    
    Lambda = np.zeros(GP_prior_cov_inv.shape)
    
    # Process preference feedback:
    num_samples = pref_data.shape[0]
    
    for i in range(num_samples):   # Go through each pair of data points
        
        data_pts = pref_data[i, :]      # Data points queried in this sample
        label = pref_labels[i]
        
        s_pos = int(data_pts[label])
        s_neg = int(data_pts[1 - label])
        
        z = (f[s_pos] - f[s_neg]) / preference_noise
        
        sigm = sigmoid(z)
        value = coeff['pref']*(-sigmoid_2nd_der(z) / sigm + (sigmoid_der(z) / sigm)**2) / (preference_noise**2)
        
        Lambda[s_pos, s_pos] += value
        Lambda[s_neg, s_neg] += value
        Lambda[s_pos, s_neg] -= value
        Lambda[s_neg, s_pos] -= value
        #lamb_i_terms = Lambda.diagonal().copy()
     
    for i in range(len(coact_idx)):   # Go through each pair of data points
        
        data_pts = coact_idx[i]      # Data points queried in this sample
        label = coact_labels[i]
        
        s_pos = data_pts[label]
        s_neg = data_pts[1 - label]
        
        z = (f[s_pos] - f[s_neg]) / preference_noise
        
        sigm = sigmoid(z)
        value = coeff['coact']*(-sigmoid_2nd_der(z) / sigm + (sigmoid_der(z) / sigm)**2) / (preference_noise**2)
        
        Lambda[s_pos, s_pos] += value
        Lambda[s_neg, s_neg] += value
        Lambda[s_pos, s_neg] -= value
        Lambda[s_neg, s_pos] -= value

    # Process human-guided feedback:
    for i in range(len(HG_points)):

        HG_region = HG_regions[i]
        region_vals = f[HG_region]
        region_max = np.max(region_vals)
        
        # Consider all points in the HG feedback region at which f is maximized
        region_argmax_idxs = HG_region[np.where(region_vals == region_max)[0]]
        
        s_neg = HG_points[i]  
     
        # Calculate value to use for updating the Hessian
        z = (region_max - f[s_neg]) / preference_noise
        sigm = sigmoid(z)
        value = coeff['direc'] *(-sigmoid_2nd_der(z) / sigm + (sigmoid_der(z) / sigm)**2) / (preference_noise**2)

        Lambda[s_neg, s_neg] += value
        
        # Consider all points in the HG feedback region at which f is maximized
        for s_pos in region_argmax_idxs:
                   
            Lambda[s_pos, s_neg] -= value
            Lambda[s_neg, s_pos] -= value
            Lambda[s_pos, region_argmax_idxs] += value
        

        
    if len(OR_points):
        OR_regions = np.array(OR_regions)
        z1 = (OR_regions[OR_labels] -  f[OR_points])/ ordinal_noise
        z2 = (OR_regions[np.subtract(OR_labels,1)] -  f[OR_points])/ ordinal_noise
        sigmz = sigmoid(z1) - sigmoid(z2)
        sigmz[sigmz == 0] = 10**-100
            
        first_i_terms = (sigmoid_2nd_der(z1)/ordinal_noise**2 -sigmoid_2nd_der(z2)/ordinal_noise**2)/sigmz
        #first_i_terms =( (1-2*sigmoid(z1))* sigmoid_der(z1)  - (1-2*sigmoid(z2))* sigmoid_der(z2)) / sigmz
        second_i_terms = ((sigmoid_der(z1)/ordinal_noise - sigmoid_der(z2)/ordinal_noise) / sigmz)**2 
        final_i_terms = - coeff['ord'] * (first_i_terms - second_i_terms)
        
        #first_i_terms = (sigmoid_2nd_der(z1)**2 -sigmoid_2nd_der(z2)**2)/(sigmz * (sigmoid_2nd_der(z1) + sigmoid_2nd_der(z2)))
        #first_i_terms =( (1-2*sigmoid(z1))* sigmoid_der(z1)  - (1-2*sigmoid(z2))* sigmoid_der(z2)) / sigmz
        #second_i_terms = ((sigmoid_der(z1)/ordinal_noise - sigmoid_der(z2)/ordinal_noise) / sigmz)**2 
        #final_i_terms = - coeff['ord'] * (first_i_terms/ordinal_noise**2 - second_i_terms)
        #lamb_copy = Lambda.diagonal().copy()
        lamb_i_terms = Lambda.diagonal().copy()
        for i in range(len(OR_points)):
            lamb_i_terms[OR_points[i]] += final_i_terms[i]  
        #lamb_i_terms[OR_points] = lamb_i_terms[OR_points] - 1/ordinal_noise**2 * (first_i_terms - second_i_terms)
        np.fill_diagonal(Lambda, lamb_i_terms)
        #print('final_i',min(final_i_terms))
        if  min(lamb_i_terms) < 0 :
            if logger:
                logger.info("Lambda term <0 ")
                logger.info('final_i' + str(final_i_terms))
                logger.info('z1: ' +str(z1))
                logger.info('z2: ' + str(z2))
                logger.info(lamb_i_terms)
            #sys.exit()
    return GP_prior_cov_inv + Lambda
    

def sigmoid(x):
    """
    Evaluates the sigmoid function at the specified value.
    Input: x = any scalar
    Output: the sigmoid function evaluated at x.
    """
    
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    """
    Evaluates the sigmoid function's derivative at the specified value.
    Input: x = any scalar
    Output: the sigmoid function's derivative evaluated at x.
    """
    return sigmoid(x)*(1-sigmoid(x))
    #return np.exp(-x) / (1 + np.exp(-x))**2

def sigmoid_2nd_der(x):
    """
    Evaluates the sigmoid function's 2nd derivative at the specified value.
    Input: x = any scalar
    Output: the sigmoid function's 2nd derivative evaluated at x.
    """
    return sigmoid(x)*((1-sigmoid(x))**2) - (sigmoid(x)**2)*(1-sigmoid(x))
    #return 2*np.exp(-2 * x)/(1 + np.exp(-x))**3 - np.exp(-x)/(1 + np.exp(-x))**2
    #return (-np.exp(-x) + np.exp(-2 * x)) / (1 + np.exp(-x))**3

def sigmoid_3rd_der(x):
    """
    Evaluates the sigmoid function's 3rd derivative at the specified value.
    Input: x = any scalar
    Output: the sigmoid function's 3rd derivative evaluated at x.
    """
    return sigmoid(x)*((1-sigmoid(x))**3) - 4*(sigmoid(x)**2)*(1-sigmoid(x))**2 + (sigmoid(x)**3)*(1-sigmoid(x)) 
    #return ( -2 * np.exp(-2 * x) + np.exp(-x)* (np.exp(-x)-1)**2 ) / (1 + np.exp(-x))**4    
    #return (-4 * np.exp(-2 * x) + np.exp(-x) + np.exp(-3*x))/(1 + np.exp(-x))**4 
    
def pref_likelihood(y,label,pref_noise):
    """
    Parameters
    ----------
    y: shape M x 2 or 1 x 2
    the second axis correspond to the estimate of x and x' for the one sampled reward function
    
    s : preference label
    pref_noise : noise in preference data

    Returns preference likelihood

    """
    y = np.array(y).T
    if len(y.shape) == 1:
        z = (y[label] - y[1 - label]) / pref_noise
    if len(y.shape) == 2:
        z = (y[:,label] - y[:,1 - label]) / pref_noise
    if np.min(sigmoid(z)) == 0: 
        print('pref = 0')
        print(z)
        return 10**-100 * np.ones(z.shape)
    return sigmoid(z)


def ord_likelihood(y,b,label,ord_noise):
    """
    Parameters
    ----------
    y : estimate of objective function
    b : ordinal threshold
    label : ordinal label
    ord_noise : ordinal noise

    Returns Ordinal likelihood
    """
    # b_ = [b[0]-5]
    # b_.extend(np.cumsum(b))
    b_ = [-5] #[-np.inf]
    b_.extend(np.cumsum(b))
    b_= np.append(b_,5)
    z1 = (b_[label] -  y)/ ord_noise
    z2 = (b_[label-1] -  y)/ ord_noise
    ord_lik = sigmoid(z1) -sigmoid(z2)
    ord_lik[ord_lik == 0] = 10**-100
#    if np.min(ord_lik) == 0: 
#         print('ord lab',label)
#         print('ord_likelihood: ',ord_lik)
    return ord_lik


