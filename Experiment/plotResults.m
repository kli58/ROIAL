function plotResults(postType,plotType,save_folder,variables_incld)

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

amberTools.graphics.fontsize(16);
amberTools.graphics.latexify;
set(gcf,'Position',[68 523 2493 305]);

c = colorbar;
c.Position = [0.919903666747026,0.123333333333333,0.006803461395476,0.815];

% saveas(f1,'3D_Posterior.png')

drawnow
% subaxis(4,6,1, 'Spacing', 0.03, 'Padding', 0, 'Margin', 0);


%% Add 01 projection


% 
% for c = 1:size(C,1)
%     dimInds = C(c,:);
%     
%     subplot(1,6,c);
% 
%     % un normalize posterior mean
%     postmean = mean_unique_all{c};
%     postmean = reshape(postmean,grid_shape(dimInds(2)),grid_shape(dimInds(1)));
%        
%     ord_label = [];
%     for i = 1:size(Y{c},1)
%         for j = 1:size(X{c},2)
%             lessThanCat = find(postmean(i,j) < ord_thresh);
%             ord_label(i,j) = lessThanCat(1) - 1;
%         end
%     end
%     hold on
% %     scatter3(reshape(X{c},1,[]),reshape(Y{c},1,[]),zeros(size(reshape(X{c},1,[]))),100,reshape(ord_label,1,[]),'filled')
%     sc{c}(2).ZData = ord_label;
% end
% 
% % save figure to save_folder
% print([save_folder,'3D_Posterior.png'],'-dpng')

% %% Plot 2D validation labels
% 
% figure();
% 
% for c = 1:size(C,1)
%     dimInds = C(c,:);
%     
%     subplot(2,3,c);
%     zlim(zlimvals);
%     
%     
%     %     grid_shape = 5*ones(1,4);
%     %     grid_shape(dimInds) = 20;
%     
%     if curMax ~= 0
%         colors = (mean_unique_all{c}-curMin)/(curMax-curMin);
%     else
%         colors = mean_unique_all{c};
%     end
%     
%     if strcmp(type,'contour')
%         colors = reshape(colors,grid_shape(dimInds(2)),grid_shape(dimInds(1)));
%         contour(X{c},Y{c},colors,'linewidth',2);
%     elseif strcmp(type,'surf')
%         colors = reshape(colors,grid_shape(dimInds(2)),grid_shape(dimInds(1)));
%         surf(X{c},Y{c},colors);
%         %         caxis(zlimvals);
%         view(2);
%         alpha 0.5
%     elseif strcmp(type,'scatter3')
%         scatter3(X{c},Y{c},Z{c},100,colors,'filled');
%     else
%         scatter(X{c},Y{c},100,colors,'filled');
%     end
%     
%     %%%%% Plot Validation Trials
%     hold on;
%     for i = 1:10
%         if ~strcmp(type,'scatter3')
%             curVal = validation_actions(i,dimInds);
%             
%             % find closest point on surface
% %             [~,curValXInd] = min(vecnorm(curVal(1)-X{c}(1,:),2,1));
% %             [~,curValYInd] = min(vecnorm(curVal(2)-Y{c}(:,1),1,2));
% %             curValColor = colors(curValYInd,curValXInd);
% 
%             if val_labels(i) == 1
%                 color = [0 0 1];
%             elseif val_labels(i) == 2
%                 color = [0 1 0];
%             elseif val_labels(i) == 3
%                 color = [0.85 0.325 0.098];
%             else
%                 color = [1 1 0];
%             end
%             hold on
%             scatter3(curVal(1),curVal(2),1,50,color, 'filled');
%             
%             % ordinal categories according to posterior
%             text(curVal(1),curVal(2),1,sprintf('%i',validation_order(i)),'color','k','VerticalAlignment','bottom','HorizontalAlignment','right');
%             
%             % ordinal categories according to user
%             text(curVal(1),curVal(2),1,sprintf('%i',val_labels(i)),'color','r','VerticalAlignment','bottom','HorizontalAlignment','left');
%         end
%     end
%     
%     xlabel(variables_incld{dimInds(1)});
%     xticks(actions{dimInds(1)}(1:2:end));
%     
%     ylabel(variables_incld{dimInds(2)});
%     yticks(actions{dimInds(2)}(1:2:end));
%     
%     xticklabels(actions{dimInds(1)}(1:2:end)); % remove every other one
%     yticklabels(actions{dimInds(2)}(1:2:end)); % remove every other one
%     
%     zlabel('Posterior Mean');
%     
%     grid on
%     axis manual
%     
%     amberTools.graphics.fontsize(14);
%     amberTools.graphics.latexify;
%     set(gcf,'Position',[395 30 1421 929]);
%     
%     % save figure to save_folder
%     print([save_folder,sprintf('Dim%i%i',dimInds(1),dimInds(2))],'-dpng')
% end
% 
% drawnow
% 
% % save figure to save_folder
% print([save_folder,'Val_Labels.png'],'-dpng')
% %% Plot 2D sampled points
% 
% % sampled_points, sampled_posterior
% 
% figure();
% 
% for c = 1:size(C,1)
%     dimInds = C(c,:);
%     
%     subplot(2,3,c);
%     zlim(zlimvals);
%     
%     
%     %     grid_shape = 5*ones(1,4);
%     %     grid_shape(dimInds) = 20;
%     
%     if curMax ~= 0
%         colors = (mean_unique_all{c}-curMin)/(curMax-curMin);
%     else
%         colors = mean_unique_all{c};
%     end
%     
%     if strcmp(type,'contour')
%         colors = reshape(colors,grid_shape(dimInds(2)),grid_shape(dimInds(1)));
%         contour(X{c},Y{c},colors,'linewidth',2);
%     elseif strcmp(type,'surf')
%         colors = reshape(colors,grid_shape(dimInds(2)),grid_shape(dimInds(1)));
%         surf(X{c},Y{c},colors);
%         %         caxis(zlimvals);
%         view(2);
%         alpha 0.5
%     elseif strcmp(type,'scatter3')
%         scatter3(X{c},Y{c},Z{c},100,colors,'filled');
%     else
%         scatter(X{c},Y{c},100,colors,'filled');
%     end
%     
%     %%%%% Plot Validation Trials
%     hold on;
%     for i = 1:size(sampled_points,1)
%         if ~strcmp(type,'scatter3')
%             
%             % plot sampled points
%             scatter3(sampled_points(:,dimInds(1)),sampled_points(:,dimInds(2)),ones(30,1),100,'k','filled');
%         
%         end
%     end
%     
%     xlabel(variables_incld{dimInds(1)});
%     xticks(actions{dimInds(1)}(1:2:end));
%     
%     ylabel(variables_incld{dimInds(2)});
%     yticks(actions{dimInds(2)}(1:2:end));
%     
%     
%     xticklabels(actions{dimInds(1)}(1:2:end)); % remove every other one
%     yticklabels(actions{dimInds(2)}(1:2:end)); % remove every other one
%     
%     zlabel('Posterior Mean');
%     
%     grid on
%     axis manual
%     
%     amberTools.graphics.fontsize(14);
%     amberTools.graphics.latexify;
%     set(gcf,'Position',[395 30 1421 929]);
%     
%     % save figure to save_folder
%     print([save_folder,sprintf('Dim%i%i',dimInds(1),dimInds(2))],'-dpng')
% end
% 
% drawnow
% 
% % save figure to save_folder
% print([save_folder,'Sampled_Actions.png'],'-dpng')
end
