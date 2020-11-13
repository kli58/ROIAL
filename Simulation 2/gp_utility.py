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
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
#sampled_idxs: have the same format as preference, current and previous samples
# posterior model needs to contain current sample nnd previous sample
def kernel(X1, X2, l=0.15, sigma_f=1):
    '''
    Isotropic squared exponential kernel. Computes 
    a covariance matrix from points in X1 and X2.
        
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        Covariance matrix (m x n). 
    '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 * sqdist / l**2 ) 


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
   
    
    R = np.zeros((M,num_features))
    for i in range(M):
    # Sample reward function from GP model posterior:
        R_i = mean + first_part @ X[i]
        R[i] = np.real(R_i)      # Store sampled reward function
       
    safe_idx = range(num_features)
    IG = []
    
   
    if method == 'IG_ucb' and saf_thresh: 
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
    
        h = - np.sum(np.sum(p * np.log2(p),axis=2),axis=1)
        p_avg = np.mean(p,axis = 0)
        H1 = - np.sum(p_avg * np.log2(p_avg))
        H2 = 1/M * np.sum(h)
        if np.isnan(H1) or np.isnan(H2): 
            logger.info(str(H1) + str(H2))
        IG.append(H1-H2)

    try:
        return [safe_idx[np.nanargmax(IG)]]
    except:
        return [random.randint(0,num_features-1)]
    

def feedback(pref_data, pref_labels, HG_points, HG_regions, GP_prior_cov_inv,
             OR_points, OR_threshold_estimate, OR_labels, ordinal_noise,
             preference_noise,coact_idx,coact_labels, r_init = [], logger = None,min_ord_thresh = -1):
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
    
    OR_regions = [-np.inf]
    OR_regions.extend(np.cumsum(OR_threshold_estimate))
    OR_regions = np.append(OR_regions,np.inf)


    # Remove any preference data recording no preference:
    pref_labels = pref_labels.flatten()
    pref_indices = np.where((pref_data[:, 0] != pref_data[:, 1]) & (pref_labels != 0.5))[0]
    pref_data = pref_data[pref_indices, :]
    pref_labels = pref_labels[pref_indices].astype(int)

    # Solve convex optimization problem to obtain the posterior mean reward 
    # vector via Laplace approximation:    
    if r_init == []:
        r_init = np.zeros(num_features)    # Initial guess

    res = minimize(preference_GP_objective, r_init, args = (pref_data, pref_labels, 
                           HG_points, HG_regions, OR_points, OR_regions, OR_labels,coact_idx,coact_labels,
                           GP_prior_cov_inv, preference_noise, ordinal_noise), 
                   method = 'L-BFGS-B', jac = preference_GP_gradient)
    
    # The posterior mean is the solution to the optimization problem:
    post_mean = res.x

    # Obtain inverse of posterior covariance approximation by evaluating the
    # objective function's Hessian at the posterior mean estimate:
    post_cov_inverse = preference_GP_hessian(post_mean,pref_data, pref_labels, 
                       HG_points, HG_regions, OR_points, OR_regions, OR_labels,coact_idx,coact_labels,
                       GP_prior_cov_inv, preference_noise, ordinal_noise,logger)

    # Calculate the eigenvectors and eigenvalues of the inverse posterior 
    # covariance matrix:
    evals, evecs = np.linalg.eigh(post_cov_inverse)

    # Invert the eigenvalues to get the eigenvalues corresponding to the 
    # covariance matrix:
    evals = 1 / evals
        
    
    # Return the model posterior:
    return {'mean': post_mean, 'cov_evecs': evecs, 'cov_evals': evals,'post_cov_inverse': post_cov_inverse,'or_noise':ordinal_noise}


#OR_regions--> b; OR_labels --> y    
def preference_GP_objective(f, pref_data, pref_labels, HG_points, HG_regions,
                            OR_points, OR_regions, OR_labels,coact_idx,coact_labels,
                            GP_prior_cov_inv, preference_noise, ordinal_noise):
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
    
    for i in range(num_samples):   # Go through each pair of data points
        data_pts = pref_data[i, :].astype('int')      # Data points queried in this sample
        label = pref_labels[i]
        z = (f[data_pts[label]] - f[data_pts[1-label]]) / preference_noise
        obj -= np.log(sigmoid(z))
        obj -= np.sum(np.log(sigmoid(z)))
        #print('processed pref')
        
    if len(OR_points):
        OR_regions = np.array(OR_regions)
        z1 = (OR_regions[OR_labels] -  f[OR_points])/ ordinal_noise
        z2 = (OR_regions[np.subtract(OR_labels,1)] -  f[OR_points])/ ordinal_noise
        obj -= np.sum(np.log(sigmoid(z1) - sigmoid(z2)))
    return obj



def preference_GP_gradient(f, pref_data, pref_labels, HG_points, HG_regions, 
                           OR_points, OR_regions, OR_labels,coact_idx,coact_labels,
                           GP_prior_cov_inv, preference_noise, ordinal_noise):
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
        
        value = (sigmoid_der(z) / sigmoid(z)) / preference_noise
        
        grad[s_pos] -= value
        grad[s_neg] += value
    
    
    if len(OR_points):
        OR_regions = np.array(OR_regions)
        z1 = (OR_regions[OR_labels] -  f[OR_points])/ ordinal_noise
        z2 = (OR_regions[np.subtract(OR_labels,1)] -  f[OR_points])/ ordinal_noise
        grad_i_terms = 1/ordinal_noise * (sigmoid_der(z1) - sigmoid_der(z2)) / (sigmoid(z1) - sigmoid(z2))
        for i in range(len(OR_points)):
            grad[OR_points[i]] += grad_i_terms[i]
    return grad

def preference_GP_hessian(f, pref_data, pref_labels, HG_points, HG_regions, 
                           OR_points, OR_regions, OR_labels,coact_idx,coact_labels,
                           GP_prior_cov_inv, preference_noise, ordinal_noise,logger):
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
        value = (-sigmoid_2nd_der(z) / sigm + (sigmoid_der(z) / sigm)**2) / (preference_noise**2)
        
        Lambda[s_pos, s_pos] += value
        Lambda[s_neg, s_neg] += value
        Lambda[s_pos, s_neg] -= value
        Lambda[s_neg, s_pos] -= value


        
    if len(OR_points):
        OR_regions = np.array(OR_regions)
        z1 = (OR_regions[OR_labels] -  f[OR_points])/ ordinal_noise
        z2 = (OR_regions[np.subtract(OR_labels,1)] -  f[OR_points])/ ordinal_noise
        sigmz = sigmoid(z1) - sigmoid(z2)
        sigmz[sigmz == 0] = 10**-100
            
        first_i_terms = (sigmoid_2nd_der(z1)/ordinal_noise**2 -sigmoid_2nd_der(z2)/ordinal_noise**2)/sigmz
        #first_i_terms =( (1-2*sigmoid(z1))* sigmoid_der(z1)  - (1-2*sigmoid(z2))* sigmoid_der(z2)) / sigmz
        second_i_terms = ((sigmoid_der(z1)/ordinal_noise - sigmoid_der(z2)/ordinal_noise) / sigmz)**2 
        final_i_terms = -  (first_i_terms - second_i_terms)
        
        lamb_i_terms = Lambda.diagonal().copy()
        for i in range(len(OR_points)):
            lamb_i_terms[OR_points[i]] += final_i_terms[i]  
  
        np.fill_diagonal(Lambda, lamb_i_terms)
        #print('final_i',min(final_i_terms))
        if  min(lamb_i_terms) < 0 :
            if logger:
                logger.info("Lambda term <0 ")
                logger.info('final_i' + str(final_i_terms))
                logger.info('z1: ' +str(z1))
                logger.info('z2: ' + str(z2))
                logger.info(lamb_i_terms)
     
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
    b_ = [-np.inf]
    b_.extend(np.cumsum(b))
    b_= np.append(b_,np.inf)
    z1 = (b_[label] -  y)/ ord_noise
    z2 = (b_[label-1] -  y)/ ord_noise
    ord_lik = sigmoid(z1) -sigmoid(z2)
    ord_lik[ord_lik == 0] = 10**-100
    return ord_lik

def safe_prob(mean,sigma,b1):
    z1 =(-b1 + mean)/sigma
    return norm.cdf(z1)
