% Experiment 3
% Compairing the training function 'trainscg', 'trainlm' and 'trainrp'.
function [ensemble_performance_data_lm, ensemble_performance_data_rp] = experiment_3(node, epoch, iterate)
	performance_function = 'mse';
	% trainscg
	ensemble_performance_data_scg = experiment_2(node, epoch, iterate, 'trainscg', performance_function);
    % trainlm
    ensemble_performance_data_lm = experiment_2(node, epoch, iterate, 'trainlm', performance_function);
    % trainrp
    ensemble_performance_data_rp = experiment_2(node, epoch, iterate, 'trainrp', performance_function);
end