#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:20:44 2020

@author: amyli
"""
from simulate_feedback import determine_ordinal_threshold
from gp_log_learning import sigmoid
import numpy as np
import matplotlib.pyplot as plt

def plot_noise_Effect(noise):
    fig = plt.figure()
    obj_values = np.arange(1,5,0.01) 
    ordinal_threshold = determine_ordinal_threshold(5, 0.2, [0.2,0.2,0.2], obj_values)
    print(ordinal_threshold)
    prob_total = []
    for obj_value in obj_values: 
        z1 = (ordinal_threshold[1:len(ordinal_threshold)] - obj_value)/noise
        z2 = (ordinal_threshold[0:len(ordinal_threshold)-1] - obj_value)/noise
        prob = sigmoid(z1) -sigmoid(z2)
        norm_prob = prob/np.sum(prob)
        #print(norm_prob)
        prob_total.append(norm_prob)
    
    prob_total = np.array(prob_total).T
    print(prob_total.shape)
    for i in range(5):
        plt.plot(obj_values,prob_total[i])
        if i < 4:
            plt.vlines(ordinal_threshold[i+1],0,1.1,colors = 'grey',linestyles = 'dotted')
        if i == 0 or i == 4:
            k = 1.03 *np.max(prob_total)
        else:
            k = 1.03 * np.max(prob_total[1]) 
        plt.text(obj_values[90*i+ 5],k, 'y = {i}'.format(i=i+1))
    #lb = np.random.choice(len(ordinal_threshold)-1, 1, p=norm_prob) 
    #print(lb)
    plt.title(r'$c_o$ = ' + str(noise),fontsize = 15)
    plt.ylim([0,1.1])
    plt.xlabel('f(x)',fontsize = 12)
    plt.ylabel('probability',fontsize = 12)
    plt.show()

noise = [0.1,0.15,0.2,0.3]
for i in noise:
    plot_noise_Effect(i)