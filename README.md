# ROIAL: Region of Interest Active Learning for Characterizing Exoskeleton Gait Preference Landscapes
The following video gives a quick introduction to the algorithm

[![Watch the video](https://i.vimeocdn.com/video/989452542_640.webp)](https://vimeo.com/473970586)


## Simulation code
A detailed description and specific values of the hyperparameter used in simulation can be found [here](https://github.com/kli58/ROIAL/blob/master/Simulation/ROIAL_hyperparameters.ipynb)

Example scripts to run the simulation can be found inside the simulation folder
- [2D Simulation](https://github.com/kli58/ROIAL/blob/master/Simulation/run_2D_simulation.ipynb) 
- [3D Simulation](https://github.com/kli58/ROIAL/blob/master/Simulation/run_3D_simulation.ipynb) 

___
## Experimental Results
The experimental results accompanying the publication "ROIAL:  Region  of  Interest  Active  Learning for  Characterizing  Exoskeleton  Gait  Preference  Landscapes" can be found in the Experiment folder. 

The main post-processing scripts that were used on the experimental data for the final plots are included in the repo. However, the results of these scripts are also included, so it is not necessary to run these scripts.
- exo_post_process_all_itera.py: Uses data_matrix.npy to update the posterior after each iteration of the experiments. The posteriors after each iteration are then saved to the mat file to exo_post_proces_all_iter.mat. 
- exo_finer_posterior: This script updates the posterior over a finer grid of points in order to generate a smoother plot of the posterior. The finer posterior along with the associated actions are saved as a mat file for each subject (finer_posterior.mat). This script requires a lot of memory and is time intensive. Thus, the output of the script is included for each subject.

The script used to plot the experimental figures is:
- mainPlottingScript.m: Script to plot the experimental figures included in the ROIAL publication as well as the animation of the posterior updating after each iteration.