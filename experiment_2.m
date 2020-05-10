% Experiment 2
% The node epoch combination with the lowest error rate is taken and
% do the random ensemble.
function [ensemble_performance_list, ensemble_performance] = experiment_2(node, epoch, iterate_count)
    load cancer_dataset; % loading the cancer dataset
    
    % Choose a Training Function
    trainFcn = 'trainscg';
    %trainFcn = 'trainlm';
    %trainFcn = 'trainbr';

    % Create a Pattern Recognition Network    
    hiddenLayerSize = node;
    net = patternnet(hiddenLayerSize, trainFcn);
    
    % ensemble_performance_list will store ensemble_node_count,
    % ensemble_average_performance and performance without ensemble.
    %ensemble_performance_list = zeros(12, 5);
    ensemble_performance_list = zeros(12, 6);
    % Ensemble classification
    index = 1;
    for ensemble_node_count = 3:2:25
        disp(ensemble_node_count);
        %ensemble_performance = zeros(iterate_count, 2);
        ensemble_performance = zeros(iterate_count, 3);
        ensemble_performance_error = zeros(iterate_count, 2);
        for i = 1:iterate_count
            ensemble_output_list = [];
            ensemble_performance_vote_list = [];
            x = cancerInputs;
            t = cancerTargets;
            net.trainParam.epochs = epoch;

            net.input.processFcns = {'removeconstantrows','mapminmax'};

            net.divideFcn = 'dividerand';  % Divide data randomly
            net.divideMode = 'sample';  % Divide up every sample
            net.divideParam.trainRatio = 50/100;
            %net.divideParam.valRatio = 25/100;
            net.divideParam.testRatio = 50/100;

            % Performance Function
            net.performFcn = 'crossentropy';  % Cross-Entropy
            %net.performFcn = 'mse';  % Mean squeared error

            % Choose Plot Functions
            net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
                'plotconfusion', 'plotroc'};
            
            for ensemble_nodes = 1:ensemble_node_count
                % calling the training function
                % rands(node, 699);
                [performance, percentErrors, y] = training_function(net, x, t);
                % ensemble output values
                ensemble_output_list = [ensemble_output_list; y];
                ensemble_performance_vote_list = [ensemble_performance_vote_list; performance];
            end
            output1 = ensemble_output_list(1:2:end, :);
            output2 = ensemble_output_list(2:2:end, :);
            
            majority_vote_output1 = mode(output1);
            majority_vote_output2 = mode(output2);
            majority_vote_output = [majority_vote_output1; majority_vote_output2];
            
            %majority performance
            majority_performance_vote = mode(ensemble_performance_vote_list);
            
            ensemble_performance_value = perform(net,t,majority_vote_output);
            tind = vec2ind(t);
            yind = vec2ind(majority_vote_output);
            %percentErrors = sum(tind ~= yind)/numel(tind);
            ensemble_performance_error_value = sum(tind ~= yind)/numel(tind);
            
            % Performance without ensemble
            [performance, percentErrors, y] = training_function(net, x, t);
            disp(performance);
            
            disp(majority_performance_vote);
            
            ensemble_performance(i, :) = [ensemble_performance_value, performance, majority_performance_vote];
            ensemble_performance_error(i, :) = [ensemble_performance_error_value, percentErrors];
        end
        
        average_ensemble_performance = sum(ensemble_performance(:,1)) / iterate_count;
        average_performance = sum(ensemble_performance(:,2)) / iterate_count;
        % average_performace of performance list
        average_performance_majority_vote = sum(ensemble_performance(:,3)) / iterate_count;
        
        average_ensemble_performance_error = sum(ensemble_performance_error(:,1)) / iterate_count;
        average_performance_error = sum(ensemble_performance_error(:,2)) / iterate_count;
        ensemble_performance_list(index, :) = [ensemble_node_count average_ensemble_performance average_performance average_ensemble_performance_error average_performance_error average_performance_majority_vote];
        index = index + 1;
    end
    % Performance graph
    figure; % Performance vs ensemble classifier
    x = ensemble_performance_list(:,1);
    y = [ensemble_performance_list(:,6) ensemble_performance_list(:,3)];
    bar(x, y), legend('Ensemble Performance', 'Performance of single network'), xlabel('Ensemble Count'), ylabel('Performance error rate')
    % performance error graph
    figure; % Performance error vs ensemble classifier
    x = ensemble_performance_list(:,1);
    y = [ensemble_performance_list(:,4) ensemble_performance_list(:,5)];
    bar(x, y), legend('Ensemble Performance Error', 'Performance error of single network'), xlabel('Ensemble Count'), ylabel('Performance Error')
end    