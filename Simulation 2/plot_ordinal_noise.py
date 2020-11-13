#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:20:44 2020

@author: amyli
"""

from gp_utility import sigmoid
import numpy as np
import matplotlib.pyplot as plt


def plot_noise_effect(noise, ordinal_threshold, obj_values):
    fig = plt.figure()

    prob_total = []
    for obj_value in obj_values: 
        z1 = (ordinal_threshold[1:len(ordinal_threshold)] - obj_value)/noise
        z2 = (ordinal_threshold[0:len(ordinal_threshold)-1] - obj_value)/noise
        prob = sigmoid(z1) -sigmoid(z2)
        norm_prob = prob/np.sum(prob)
        prob_total.append(norm_prob)
    
    prob_total = np.array(prob_total).T
    for i in range(5):
        plt.plot(obj_values,prob_total[i])
        if i < 4:
            plt.vlines(ordinal_threshold[i+1],0,1.1,colors = 'grey',linestyles = 'dotted')
        if i == 0 or i == 4:
            k = 1.03 *np.max(prob_total)
        else:
            k = 1.03 * np.max(prob_total[1]) 
        plt.text(obj_values[90*i+ 5],k, 'y = {i}'.format(i=i+1))

    plt.title(r'$c_o$ = ' + str(noise),fontsize = 15)
    plt.ylim([0,1.1])
    plt.xlabel('f(x)',fontsize = 12)
    plt.ylabel('P(y|f(x))',fontsize = 12)
    plt.show()
