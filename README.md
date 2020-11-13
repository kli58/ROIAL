# ROIAL: Region of Interest Active Learning for Characterizing Exoskeleton Gait Preference Landscapes
The following video gives a quick introduction to the algorithm

[![Watch the video](https://i.vimeocdn.com/video/989452542_640.webp)](https://vimeo.com/473970586)

[Here](https://github.com/kli58/ROIAL/blob/master/Simulation/run_2D_simulation.ipynb) is an example that shows how to run 2D simulation

To understand some of the hyperparameters better, check [this](https://github.com/kli58/ROIAL/blob/master/Simulation/ROIAL_hyperparameters.ipynb)

___
## Experimental Results
The experimental results accompanying the publication "ROIAL:  Region  of  Interest  Active  Learning for  Characterizing  Exoskeleton  Gait  Preference  Landscapes" can be found in the Experiment folder. The scripts to run are the following:
- mainPlottingScript.m: Script to plot the experimental figures included in the ROIAL publication as well as the animation of the posterior updating after each iteration.
- exo_post_process_all_itera.py: Uses data_matrix.npy to update the posterior after each iteration of the experiments. The posteriors for each iteration are then saved to the mat file to exo_post_proces_all_iter.mat. (this does not need to be done as the mat file for each subject is included in the repository but the code is included anyways)
