function iterations(save_folder)

if save_folder(end) ~= '/'
    save_folder = [save_folder, '/'];
end

% load saved information from subject folder
load([save_folder,'exo_post_process_all_iter.mat'],'posterior_mean_whole');
load([save_folder,'active_GP_opt_output.mat'],'points_to_sample','sampled_flt_idx');
load([save_folder,'ExperimentInfo.mat'],'tempExpInfo');
ord_labels = tempExpInfo(:,4);

% set up figure handles
figure(); 
t = tiledlayout(1,5);
for i = 1:5
    h(i) = nexttile();    
end

% go through each iteration and plot updated posterior
for i = 1:size(posterior_mean_whole,1)
    
    title(t,sprintf('Iteration %i',i+1),'FontWeight','Bold');
    curPosterior = posterior_mean_whole(i,:);
    Plotting.slices(save_folder,curPosterior,h);
        
    % add compared actions with their ordinal labels to iteration plot
    for j = 0:1
        temp = points_to_sample(sampled_flt_idx(i+j)+1,:);
        
        switch temp(4)
            case 10.5
                curplot = h(1);
            case 11.5
                curplot = h(2);
            case 12.5
                curplot = h(3);
            case 13.5
                curplot = h(4);
            case 14.5
                curplot = h(5);
        end
        
        scatter3(curplot,temp(1),temp(2),temp(3),500,'k','LineWidth',2);
        text(curplot,temp(1),temp(2),temp(3),num2str(ord_labels{i+j+1}));
        
    end
    
    % Save plot as png
    fprintf(sprintf('Iteration %i/%i Saved \n',i+1,size(posterior_mean_whole,1)+1));
    if i+1 < 10
        name = [save_folder 'iter_0' num2str(i+1)];
    else
        name = [save_folder 'iter_' num2str(i+1)];
    end
    saveas(gcf,[name '.png']);
end

end