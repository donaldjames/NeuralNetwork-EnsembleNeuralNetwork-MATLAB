% Experiment 2
% The node epoch combination with the lowest error rate is taken and
% do the random ensemble.
function ensemble_performance_list = experiment_2(hiddenLayerSize, epoch, iterate_count, train_function)
    load cancer_dataset; % loading the cancer dataset
    
    trainFcn = train_function; % Training function
    
    % Create a Pattern Recognition Network    
    % First network(For ensemble NN)
    net1 = patternnet(hiddenLayerSize, trainFcn);
    net1.trainParam.epochs = epoch;
    net1.input.processFcns = {'removeconstantrows','mapminmax'};
    net1.performFcn = 'mse';  % Mean squeared error
    %net1.performFcn = 'crossentropy';  % Cross-Entropy
    net1.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
                'plotconfusion', 'plotroc'}; % plotting the listed graphs while training the model
    
    % Second network(without ensemble NN)
    net2 = patternnet(hiddenLayerSize, trainFcn);
    net2.trainParam.epochs = epoch;
    net2.input.processFcns = {'removeconstantrows','mapminmax'};
    %net2.performFcn = 'crossentropy';  % Cross-Entropy        
    net2.performFcn = 'mse';  % Mean squeared error
    net2.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        'plotconfusion', 'plotroc'}; % plot functions         
    
    % ensemble_performance_list will store ensemble_node_count,
    % ensemble_average_performance and performance without ensemble.
    ensemble_performance_list = zeros(12, 3);
    % Ensemble classification
    index = 1;
    for ensemble_node_count = 3:2:25
        disp(ensemble_node_count);
        ensemble_performance = [];
        for i = 1:iterate_count
            ensemble_output_list = [];
            
            x = cancerInputs;
            t = cancerTargets;
            
            % First network. For ensemble NN
            net1.divideFcn = 'dividerand';  % Divide data randomly
            net1.divideMode = 'sample';  % Divide up every sample
            net1.divideParam.trainRatio = 50/100;
            net1.divideParam.testRatio = 50/100;
            
            % Second network. without ensemble
            net2.divideFcn = 'dividerand';  % Divide data randomly
            net2.divideMode = 'sample';  % Divide up every sample
            net2.divideParam.trainRatio = 50/100;
            net2.divideParam.testRatio = 50/100;
            
            for ensemble_nodes = 1:ensemble_node_count
                % Training the network
                % rands(node, 699);
                % randomw = rands(8,699);
				% net1.iw(randomw);
                [net1, tr] = train(net1,x,t);
                y = net1(x);
                ensemble_output_list = [ensemble_output_list; y];
            end
            output1 = ensemble_output_list(1:2:end, :);
            output2 = ensemble_output_list(2:2:end, :);
            
            majority_vote_output1 = mode(output1);
            majority_vote_output2 = mode(output2);
            majority_vote_output = [majority_vote_output1; majority_vote_output2];
            
            ensemble_performance_value = perform(net1,t,majority_vote_output);
            tind = vec2ind(t);
            yind = vec2ind(majority_vote_output);
            %percentErrors = sum(tind ~= yind)/numel(tind);
            %ensemble_performance_error_value = sum(tind ~= yind)/numel(tind);
            
            % Performance without ensemble
            % rands(8,699);
            [net2, tr] = train(net2,x,t);
            y = net2(x);
            performance = perform(net2,t,y);
            
            % storing the ensemble performance along with individual
            % performance onto ensemble_performance
            ensemble_performance = [ensemble_performance; ensemble_performance_value, performance];
        end
        
        average_ensemble_performance = (sum(ensemble_performance(:,1)) / iterate_count);
        average_performance = (sum(ensemble_performance(:,2)) / iterate_count);
        
        ensemble_performance_list(index, :) = [ensemble_node_count average_ensemble_performance average_performance];
        index = index + 1;
    end
    % Performance graph
    figure; % Performance vs ensemble classifier
    x = ensemble_performance_list(:,1);
    y = [ensemble_performance_list(:,2) ensemble_performance_list(:,3)];
    bar(x, y), legend('Ensemble Performance', 'Without ensemble performance'), xlabel('Ensemble Classifier'), ylabel('Performance Error')
    title(train_function);
% end    