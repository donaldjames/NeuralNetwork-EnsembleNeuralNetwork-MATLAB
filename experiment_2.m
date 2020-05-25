% Experiment 2
% The node epoch combination with the lowest error rate is taken and
% do the random ensemble.
function ensemble_performance_list = experiment_2(hiddenLayerSize, epoch, iterate_count, train_function, performance_function)
    load cancer_dataset; % loading the cancer dataset
    
    trainFcn = train_function; % Training function
    
    % General network definitions   
	net_definition = patternnet(hiddenLayerSize, trainFcn);
	net_input_processfcns = {'removeconstantrows','mapminmax'};
	net_performfcn = performance_function; % performance function
	% plotting the listed graphs while training the model
	net_plot_functions = {'plotperform','plottrainstate', ...
						'ploterrhist', 'plotconfusion', 'plotroc'}; 
    
    % ensemble_performance_list will store ensemble_node_count,
    % ensemble_average_performance and performance without ensemble.
    ensemble_performance_list = zeros(12, 3);
    index = 1;
    for ensemble_node_count = 3:2:25
        disp(ensemble_node_count);
        ensemble_performance = [];
    
        % First network(For ensemble NN)
        net1 = net_definition;
        net1.trainParam.epochs = epoch;
        net1.input.processFcns = net_input_processfcns;
        net1.performFcn = net_performfcn;
        net1.plotFcns = {};
    
        % Second network(without ensemble NN)
        net2 = net_definition;
        net2.trainParam.epochs = epoch;
        net2.input.processFcns = net_input_processfcns;
        net2.performFcn = net_performfcn;  
        net2.plotFcns = {};
        
        for i = 1:iterate_count
			% list to store ensemble output values
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
            
			% Ensemble with 'ensemble_nodes' individual classifiers
            for ensemble_nodes = 1:ensemble_node_count
                % Training the network
                net1.inputWeights{1,''}.initFcn = rands(9,699);
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
            
            % Performance without ensemble
            net2.inputWeights{1,''}.initFcn = rands(9,699);
            [net2, tr] = train(net2,x,t);
            y = net2(x);
            performance = perform(net2,t,y);
            
            % storing the ensemble performance along with individual
            % performance to ensemble_performance list
            ensemble_performance = [ensemble_performance; ensemble_performance_value, performance];
        end
        
        average_ensemble_performance = (sum(ensemble_performance(:,1)) / iterate_count);
        average_performance = (sum(ensemble_performance(:,2)) / iterate_count);
        
        ensemble_performance_list(index, :) = [ensemble_node_count average_ensemble_performance average_performance];
        index = index + 1;
    end
end    