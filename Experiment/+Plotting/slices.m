function slices(save_folder,posterior_mean,h)

screensize = get( 0, 'Screensize' );

if nargin < 1
    save_folder = uigetdir(pwd);
    save_folder = [save_folder, '/'];
end

if save_folder(end) ~= '/'
    save_folder = [save_folder, '/'];
end


%%% CURRENTLY ONLY WORKS FOR SCATTER3
type = 'scatter3';


%%% Plots dimensions 1,2,3 and slices of 4
load([save_folder,'active_GP_params'],'lower_bounds','discretization','upper_bounds');
for i = 1:4
    actions{i} = lower_bounds(i):discretization(i):upper_bounds(i);
end


load([save_folder,'active_GP_params'],'grid_shape');
load([save_folder,'active_GP_opt_output'],'points_to_sample');
load([save_folder,'exo_post_process_all_iter'],'posterior_mean_whole');
load([save_folder,'ExperimentInfo'],'val_labels','validation_actions','validation_order');

%%%% Meant for state_dim = 4;
[num_points, state_dim] = size(points_to_sample);

if nargin < 2
    load([save_folder,'exo_post_process'],'posterior_mean');
end

% normalize by min and max of current posterior
maxAll = max(reshape(posterior_mean_whole,1,[]));
minAll = min(reshape(posterior_mean_whole,1,[]));


for i = 1:length(actions{4}) %number of points in 4th dimension
    
    % open a new figure
%     subplot(handles{i});

    hold(h(i), 'off');
    curAction4 = actions{4}(i);
    
    slicePointInds = find(points_to_sample(:,4) == curAction4);
    

    colors = posterior_mean(slicePointInds);
    
    if strcmp(type,'scatter3')
        scatter3(h(i),points_to_sample(slicePointInds,1),points_to_sample(slicePointInds,2),points_to_sample(slicePointInds,3),100,colors,'filled','MarkerFaceAlpha', 0.4);
        title(h(i),sprintf('Pelvis Pitch: %2.2f',curAction4));
    end
    
    % set axis limits
    xlim(h(i),actions{1}([1,end]));
    ylim(h(i),actions{2}([1,end]));
    zlim(h(i),actions{3}([1,end]));
    
    % normalize color scale across all plots
    caxis(h(i),[minAll maxAll]); 
    
    % axis settings
    axis(h(i), 'square');
    axis(h(i), 'manual');
    
    % axis labels
    xlabel(h(i),'Step Length');
    ylabel(h(i),'Step Duration');
    zlabel(h(i),'Pelvis Roll');
    
    % set grid on and hold on for action labels
    grid(h(i),'on');
    hold(h(i),'on');
    
    % manually set figure position (may need to change for different
    % monitors)
    set(gcf,'Position',[173 303 2221 481]);
    
    % If you have amber_matlab_toolbox:
%     amberTools.graphics.fontsize(10);
%     amberTools.graphics.latexify;
    
end

% add colorbar
c = colorbar(h(end));
caxis([minAll maxAll]);

drawnow

end
