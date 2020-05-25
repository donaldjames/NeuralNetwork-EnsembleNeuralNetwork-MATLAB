%Experiment 1
iterate = 30;
[node, epoch, per_list] = experiment_1(iterate);

% Experiment 2
train_fun = 'trainscg';
performance_func = 'crossentropy';
ensemble_performance_data_scg = experiment_2(node, epoch, iterate, train_fun, performance_func);
% Plotting the graph
figure; % Performance vs ensemble classifier
x = ensemble_performance_data_scg(:,1);
y = ensemble_performance_data_scg(:,2:3);
bar(x, y), legend('Ensemble Error', 'Individual Classifier Error'), xlabel('Ensemble (classifier count)'), ylabel('Error')
title(train_fun);

% Experiment 3
[ensemble_performance_data_scg, ensemble_performance_data_lm, ensemble_performance_data_rp] = experiment_3(node, epoch, iterate);
figure; % training function comparison graph
x = ensemble_performance_data_scg(:,1);
y_scg = ensemble_performance_data_scg(:,2:3);
y_lm = ensemble_performance_data_lm(:,2:3);
y_rp = ensemble_performance_data_rp(:,2:3);
bar(x, [y_scg y_lm y_rp]), xlabel('Ensemble (classifier count)'), ylabel('Error')
legend("SCG ensemble", "SCG individual", "LM ensemble", "LM individual", "RP ensemble", "RP individual"),
title('Training Function Performance');