# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 13:09:24 2020

@author: amykj
"""

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import scipy.io as io
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from plot_individual import norm_objective_data


def plot_3d_function(D,run_num,iter_list, save_folder, obj_folder, num_pts_each_dim = (20,20,20), save = False):
    filename = save_folder + 'dim' + str(D) + '_run_' + str(run_num)  +'.mat'  
    sub_file = save_folder + 'dim' + str(D) + '_run_' + str(run_num)  +'_post_process.mat'
    posterior_list =  io.loadmat(sub_file)['posterior_mean']
    
    objective_values = io.loadmat(filename)['norm_data']
    objective_values = norm_objective_data(objective_values)
    obj_file = obj_folder + str(run_num) + '.mat'
    obj_val = io.loadmat(obj_file)['sample']
    points_to_sample = io.loadmat(obj_file)['points_to_sample']

    obj_avg = np.mean(obj_val,axis = 2)
    obj_avg_norm = norm_objective_data(obj_avg)
    num_points = points_to_sample.shape[0]
    idx_reshape = np.unravel_index(np.arange(num_points), num_pts_each_dim)
    pos_list = []
    for i in range(len(posterior_list)):
        pts_reshape = np.zeros(num_pts_each_dim)
        pts_reshape[idx_reshape] = posterior_list[i]
    
        pos_avg = np.mean(pts_reshape,axis = 2)
        pos_avg_norm = norm_objective_data(pos_avg)
    
        pos_list.append(pos_avg_norm)
        
    #io.savemat('simulation_posterior_0.5_10_iter.mat', { 'objective':obj_avg_norm,'posterior': pos_list,'iter_list':iter_list})
  
    for i in range(len(iter_list)):
 
        pos_avg_norm= pos_list[i]
        trial_num = iter_list[i]
     
        
        fig = plt.figure(figsize = (8.5,4))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        # ax = fig.gca(projection='3d')
        
        # Make data.
        X = np.linspace(0, 1,20)
        Y = np.linspace(0, 1,20)
        X, Y = np.meshgrid(X, Y)
       

        # Plot the surface.
        color_map = plt.cm.get_cmap('YlGnBu')
        surf = ax1.plot_surface(X, Y, obj_avg_norm, cmap=color_map,
                                linewidth=0, antialiased=False)
        
        # Customize the z axis.
        ax1.set_xlim(0, 1.0)
        ax1.set_ylim(0, 1.0)
        ax1.set_zlim(0,1.0)
        ax1.set_xticks([0,0.5,1])
        ax1.set_yticks([0,0.5,1])
        ax1.set_zticks([0,0.5,1])
        ax1.set_xlabel('$x_1$')
        ax1.set_ylabel('$x_2$')
        ax1.set_zlabel('f(x)')
        
        ax1.set_title('objetive function',fontsize = 12)

        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        # ax = fig.gca(projection='3d')
        
        
        #'YlGnBu'
        # Plot the surface.
        surf = ax2.plot_surface(X, Y, pos_avg_norm, cmap= color_map,
                                linewidth=0, antialiased=False)
        
        # Customize the z axis.
        ax2.set_title('posterior mean',fontsize = 12)
        ax2.set_xlim(0, 1.0)
        ax2.set_ylim(0, 1.0)
        ax2.set_zlim(0,1.0)
        ax2.set_xticks([0,0.5,1])
        ax2.set_yticks([0,0.5,1])
        ax2.set_zticks([0,0.5,1])
        
        
        ax2.set_xlabel('$x_1$')
        ax2.set_ylabel('$x_2$')
        ax2.set_zlabel('f(x)')
        
  
        # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        
        plt.suptitle('Iteration ' + str(trial_num),fontsize = 15)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        fig.colorbar(surf, cax=cbar_ax,shrink = 0.6)
        plt.show()
        if save:
            fig.savefig(save_folder + 'iter' + str(trial_num) + '.png')

    
    




