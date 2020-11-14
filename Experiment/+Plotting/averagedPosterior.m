function averagedPosterior(postType,plotType,save_folder,variables_incld)

if nargin < 2
    type = 'surf';
else
    type = plotType;
end

if nargin < 3
    save_folder = uigetdir(pwd);
    save_folder = [save_folder, '/'];
end

if save_folder(end) ~= '/'
    save_folder = [save_folder, '/'];
end

switch postType
    case 'normal'
        load([save_folder,'active_GP_opt_output'],'points_to_sample');
        load([save_folder,'exo_post_process'],'posterior_mean');
%         grid_shape = [10,7,5,5];
        load([save_folder,'active_GP_params'],'grid_shape');

    case 'fine'
        load_name = sprintf('finer_posterior');
%         load_name = sprintf('exo_finer_posterior'); %[20,20,20,10] but
%         doesn't have new_points so plotting is messed up
        load([save_folder,load_name],'new_points','posterior_mean_whole');
        points_to_sample = new_points;
        points_to_sample(1:30,:) = [];
        sampled_points = new_points(1:30,:);
        posterior_mean = posterior_mean_whole;
        posterior_mean(1:30) = [];
        sampled_posterior = posterior_mean_whole(1:30);
        grid_shape = [20,20,10,10];
    otherwise
        error('posterior type must be set to either "normal" or "fine"');
end



load([save_folder,'active_GP_params'],'lower_bounds','discretization','upper_bounds');
for i = 1:4
    actions{i} = lower_bounds(i):discretization(i):upper_bounds(i);
end

load([save_folder,'ExperimentInfo'],'val_labels','validation_actions','validation_order');



paramfile = [save_folder 'active_GP_opt_output.mat'];
load(paramfile,'ord_thresh_est');
ord_thresh = [-Inf cumsum(ord_thresh_est) Inf];


%% Save data

state_dim = 4;

if nargin < 4
    variables_incld = cell(1,state_dim);
    variables_incld = {'SL','SD','PR','PP'};
else
    for i = 1:state_dim
        variables_incld{i} = strrep(variables_incld{i},'_',' ');
    end
end


% all combinations of parameters based on plotType
if strcmp(type,'scatter3')
    C = nchoosek(1:state_dim,3);
else
    C = nchoosek(1:state_dim,2);
end

% History of max and min colorbars
curMax = -inf;
curMin = inf;

% plot combinations of 2 dimensions at a time
for c = 1:size(C,1)
    
    % current combination to plot
    dimInds = C(c,:);
    
    %     grid_shape = 5*ones(1,4);
    %     grid_shape(dimInds) = 20;
    
    % reduced subset to plot (2 dimensions)
    reduced_subset = points_to_sample(:,dimInds);
    [unique_subset,all2unique,unique2all] = unique(reduced_subset,'rows');
    % unique_subset = reduced_subset(all2unique)
    % reduced_subset = unique_subset(unique2all)
    
    % remove the plotted dimensions from the dimensions to mean over
    mean_unique = zeros(size(unique_subset,1),1);
    
    % go through all unique points
    for j = 1:length(all2unique)
        
        % take the mean over all repeating entries
        mean_inds = (unique2all == j);
        mean_unique(j) = mean(posterior_mean(mean_inds));
    end
    
    if max(mean_unique) > curMax
        curMax = max(mean_unique);
    end
    if min(mean_unique) < curMin
        curMin = min(mean_unique);
    end
    
    if strcmp(type,'scatter')
        X{c} = unique_subset(:,1);
        Y{c} = unique_subset(:,2);
        mean_unique_all{c} = mean_unique;
    elseif strcmp(type,'scatter3')
        X{c} = unique_subset(:,1);
        Y{c} = unique_subset(:,2);
        Z{c} = unique_subset(:,3);
        mean_unique_all{c} = mean_unique;
    else
        X{c} = reshape(unique_subset(:,1),grid_shape(dimInds(2)),grid_shape(dimInds(1)));
        Y{c} = reshape(unique_subset(:,2),grid_shape(dimInds(2)),grid_shape(dimInds(1)));
        mean_unique_all{c} = mean_unique;
    end
end

zlimvals = [curMin, curMax];

%% PLOT 3D Posterior
f1 = figure();
clf(f1);

for c = 1:size(C,1)
    dimInds = C(c,:);
    
    subplot(1,6,c);
    
    if curMax ~= 0
        colors = (mean_unique_all{c}-curMin)/(curMax-curMin);
    else
        colors = mean_unique_all{c};
    end
    
    %     grid_shape = 5*ones(1,4);
    %     grid_shape(dimInds) = 20;
    
    if strcmp(type,'contour')
        colors = reshape(colors,grid_shape(dimInds(2)),grid_shape(dimInds(1)));
        contour(X{c},Y{c},colors,'linewidth',2);
        %         caxis(zlimvals);
    elseif strcmp(type,'surf')
        zlim([0,1]);
        colors = reshape(colors,grid_shape(dimInds(2)),grid_shape(dimInds(1)));
        sc{c} = surf(X{c},Y{c},colors,'EdgeAlpha',0.2);
        
        hold on;
        [~,h] = contourf(X{c},Y{c},colors);
        
        hh = get(h,'Children');
        for i=1:numel(hh)
            zdata = ones(size( get(hh(i),'XData') ));
            set(hh(i), 'ZData',-10*zdata)
        end
        
        % find O1 region
        levels = (ord_thresh(2:end)-curMin)/(curMax-curMin);
        
%         h.LevelList = levels;
        alpha 0.8
        
%         view([-44.10434813336181,17.067212586203361])
    elseif strcmp(type,'scatter3')
        scatter3(X{c},Y{c},Z{c},100,colors,'filled');
        %         caxis(zlimvals);
    else
        scatter(X{c},Y{c},100,colors,'filled');
    end
    
    
    xlabel(variables_incld{dimInds(1)}, 'VerticalAlignment','middle', 'HorizontalAlignment','center');
    xticks(actions{dimInds(1)});
    
    ylabel(variables_incld{dimInds(2)}, 'VerticalAlignment','middle', 'HorizontalAlignment','center');
    yticks(actions{dimInds(2)});
    
    xlabels = string(actions{dimInds(1)}); 
    ylabels = string(actions{dimInds(2)}); 
    xlabels(2:end-1) = '';
    ylabels(2:end-1) = '';
    xticklabels(xlabels);
    yticklabels(ylabels);
    
    xlim([min(actions{dimInds(1)}),max(actions{dimInds(1)})])
    ylim([min(actions{dimInds(2)}),max(actions{dimInds(2)})])
    zlim([0,1]);
    axis square
    view(3)
    
    % manually add last entry for even bin dimensions
%     if dimInds(1) == 1
%         xticks(actions{dimInds(1)}([1:2:end,end]));
%         xticklabels(actions{dimInds(1)}([1:2:end,end])); % remove every other one
%     elseif dimInds(2) == 1
%         yticks(actions{dimInds(2)}([1:2:end,end]));
%         yticklabels(actions{dimInds(2)}([1:2:end,end])); % remove every other one
%     end
    
    zlabel('Post. Mean');
    zticks([0,0.5,1]);
    grid on
    
    
end

% use if you have amber_matlab_toolbox
% amberTools.graphics.fontsize(16);
% amberTools.graphics.latexify;

set(gcf,'Position',[68 523 2493 305]);

c = colorbar;
c.Position = [0.919903666747026,0.123333333333333,0.006803461395476,0.815];

drawnow

end
