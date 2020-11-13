%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: This script generates all the experimental plots included in
%   ICRA2021 publication
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

folderNames = {'sub1','sub2','sub3'};

%% plot static final posterior plots for each subject
for sub = 1:3
    save_folder = ['Results/',folderNames{sub}];
    plotResults('fine','surf',save_folder);
end

%% plot updating posteriors
for sub = 2:3
    save_folder = ['Results/',folderNames{sub}];
    plotIterations(save_folder);
end