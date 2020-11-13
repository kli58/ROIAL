# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:55:31 2019

Sample from a Gaussian process prior.

@author: Ellen
"""


import numpy as np
import matplotlib.pyplot as plt

def sample_GP_prior(variance, lengthscale, noise_var, num_GP_samples, sample_points, mean):
    # GP prior covariance:
    num_sample_points = len(sample_points)
    cov = np.empty((num_sample_points, num_sample_points))
    
    for i in range(num_sample_points):
        
        x1 = sample_points[i]
        
        for j in range(i + 1):
            
            x2 = sample_points[j]
            
            RBF_val = variance * np.exp(-(x1 - x2)**2 / (2 * lengthscale**2))
            
            if i == j:
                
                cov[i, i] = RBF_val + noise_var
                
            else:
                
                cov[i, j] = RBF_val
                cov[j, i] = RBF_val
        
                
    # Draw several samples from the GP:
    samples = []
    
    for i in range(num_GP_samples):
        samples.append(np.random.multivariate_normal(mean, cov))
        
    # Plot the GP samples:
    plt.figure()
    
    for i in range(num_GP_samples):
        
        plt.plot(sample_points, samples[i])
        
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Samples from GP Prior')
