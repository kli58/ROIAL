# -*- coding: utf-8 -*-
"""
Functions for obtaining preference and coactive feedback. They are used by the
2D objective function simulations.
"""

import numpy as np
from gp_utility import sigmoid

def hartmann_objective_function(pt, dim):
    """3d or 6d Hartmann test function
        input bounds:  0 <= xi <= 1, i = 1..6
        global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
        min function value = -3.32237
    """
    alpha = [1.00, 1.20, 3.00, 3.20]
    if dim == 3:
        A = np.array([[3.0, 10.0, 30.0],
                        [0.1, 10.0, 35.0],
                        [3.0, 10.0, 30.0],
                        [0.1, 10.0, 35.0]])
        P = 0.0001 * np.array([[3689, 1170, 2673],
                                [4699, 4387, 7470],
                                [1090, 8732, 5547],
                                [381, 5743, 8828]])
    elif dim == 6:
        A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                    [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                    [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                    [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
        P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                            [2329, 4135, 8307, 3736, 1004, 9991],
                            [2348, 1451, 3522, 2883, 3047, 6650],
                            [4047, 8828, 8732, 5743, 1091, 381]])
    else:
        raise ValueError('Hartmann function should have either 3 or 6 dimensions')

    external_sum = 0
    for i in range(4):
        internal_sum = 0
        for j in range(dim):
            internal_sum += A[i, j] * (pt[j] - P[i, j]) ** 2
        external_sum += alpha[i] * np.exp(-internal_sum)

    return external_sum

def get_objective_value(point_idx, objective):
    """
    Evaluates the objective function at x.
    
    Inputs: 1) point_idx is the index of the point at which to evaluate the
               objective. (This is the point's index in the prior covariance 
               matrix, as well as the row index in the points_to_sample array.)
            2) n-by-m NumPy array of the objective function value in each
               grid cell. (n = m = 30 for ICRA simulations.)
    
    Output: scalar value of the objective function at the queried point.
    """
    
    return objective.flatten()[point_idx]


        
def get_preference(pt1, pt2, objective,add_GP = False,noise = 0):
    """
    Obtain a preference between two points by preferring whichever point has a 
    higher objective function value; break ties randomly.
    
    NOTE: this differs from the get_gait_preference function used by the 
    compass-gait biped simulations, in that higher objective function values 
    are preferred, rather than lower ones.
    
    Inputs: pt1 and pt2 are the two points to be compared. Both points should 
    be arrays with length equal to the dimensionality of the input space (i.e.,
    2D for ICRA simulations).    
    
    Output: 0 = pt1 preferred; 1 = pt2 preferred.
    """
#    if fcn == 'hartmann-3d':
#        obj1 = hartmann_objective_function(pt1, dim=3)
#        obj2 = hartmann_objective_function(pt2, dim=3)
#    else:
    if add_GP:
        obj1 = objective[pt1]
        obj2 = objective[pt2]
    else:
        obj1 = get_objective_value(pt1, objective)
        obj2 = get_objective_value(pt2, objective)
    if noise:
        prob = sigmoid((obj1 - obj2)/noise)
#        print('generate noisy feedback')
        return np.random.choice(2, 1, p=[prob,1-prob])[0]
    else:
        if obj2 > obj1:
            return 1
        elif obj1 > obj2:
            return 0
        else:
            return np.random.choice(2)

def determine_ordinal_threshold(num_category, b1_val, delta_t,objective_function, perct = True):
    b = np.zeros(num_category+1)
    b[0] = np.min(objective_function)
    b[num_category] = np.max(objective_function)
    if perct:
        b[1] = np.percentile(objective_function, 100*b1_val)
        val = b1_val
        for i in range(2,num_category):
            print(delta_t[i-2])
            print(val)
            val = val + delta_t[i-2]
            b[i] = np.percentile(objective_function,100*val)
    else:
        b[1] = b1_val * (np.max(objective_function) - np.min(objective_function)) + np.min(objective_function)
        if len(delta_t) > 1:
            delta = np.array(delta_t) * (np.max(objective_function) - np.min(objective_function)) 
            for i in range(2,num_category):
                b[i] = b[i-1] + delta[i-2]
        else:
            delta = delta_t * (np.max(objective_function) - np.min(objective_function)) 
            for i in range(2,num_category):
                b[i] = b[i-1] + delta
    return b
'''
Ordinal_threshold: b
Objective : f(x) 
'''
def get_ordinal_feedback(point_idx,ordinal_threshold, objective,add_GP = False,noise = 0):
    if add_GP:
        obj_value = objective[point_idx]
    else:
        obj_value = get_objective_value(point_idx, objective)
    if noise:
#        print('generate noisy feedback')
        z1 = (ordinal_threshold[1:len(ordinal_threshold)] - obj_value)/noise
        z2 = (ordinal_threshold[0:len(ordinal_threshold)-1] - obj_value)/noise
        prob = sigmoid(z1) -sigmoid(z2)
        norm_prob = prob/np.sum(prob)
        lb = np.random.choice(len(ordinal_threshold)-1, 1, p=norm_prob) 
        #print(lb)
        return lb[0] + 1
    else:
        for i in range(1,len(ordinal_threshold)):
            if obj_value <= ordinal_threshold[i]:
                obj_label = i
                break
        if 'obj_label' in locals():
            return obj_label
        else:
            return i



