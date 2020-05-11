% Experiment 1
% setting the nodes [2 8 32] and epochs [4 8 16 32 64] and running the
% 15 combinations 'iterate_count' of times. The average error rate,
% performance mean and standard deviation is calculated and stored into a
% multidimensional array for the ease of calculations
function [node, epoch, per_list] = experiment_1(iterate_count)
    load cancer_dataset;
    
    performance_list2 = zeros(450, 3);

    trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

    % Create a Pattern Recognition Network
    nodes = [2 8 32];
    per_list = zeros(15, 6);
    iter_counter = 1;
    node_epoch_index = 1;
    for node_index = 1:length(nodes)
        hiddenLayerSize = nodes(node_index);
        net = patternnet(hiddenLayerSize, trainFcn);

        epoch_set = [ 4 8 16 32 64];
        for epoch_value_index = 1:length(epoch_set)
            performance_per_iter = zeros(iterate_count, 1);
            performance_error_per_iter = zeros(iterate_count, 1);
            for i = 1:iterate_count
                x = cancerInputs;
                t = cancerTargets;
                net.trainParam.epochs = epoch_set(epoch_value_index);

                net.input.processFcns = {'removeconstantrows','mapminmax'};

                net.divideFcn = 'dividerand';  % Divide data randomly
                net.divideMode = 'sample';  % Divide up every sample
                net.divideParam.trainRatio = 50/100;
                net.divideParam.testRatio = 50/100;

                % Performance Function
                net.performFcn = 'crossentropy';  % Cross-Entropy

                % calling the training function
                % [performance, percentErrors, y] = training_function(net, x, t);
                [net, tr] = train(net,x,t);
                y = net(x);
                e = gsubtract(t,y);
                performance = perform(net,t,y);
                tind = vec2ind(t);
                yind = vec2ind(y);
                percentErrors = sum(tind ~= yind)/numel(tind);

                % Recalculate Training, Validation and Test Performance
                trainTargets = t .* tr.trainMask{1};
                %valTargets = t .* tr.valMask{1};
                testTargets = t .* tr.testMask{1};
                trainPerformance = perform(net,trainTargets,y)
                %valPerformance = perform(net,valTargets,y)
                testPerformance = perform(net,testTargets,y)

                % Storing the performance of each run to performance_list
                performance_list2(iter_counter, :) = [testPerformance nodes(node_index) epoch_set(epoch_value_index)];
                iter_counter = iter_counter + 1;
                performance_per_iter(i, :) = performance;
                performance_error_per_iter(i, :) = testPerformance;
                

                if (false)
                    % Generate MATLAB function for neural network for application
                    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
                    % tools, or simply to examine the calculations your trained neural
                    % network performs.
                    genFunction(net,'myNeuralNetworkFunction');
                    y = myNeuralNetworkFunction(x);
                end
                if (false)
                    % Generate a matrix-only MATLAB function for neural network code
                    % generation with MATLAB Coder tools.
                    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
                    y = myNeuralNetworkFunction(x);
                end
                if (false)
                    % Generate a Simulink diagram for simulation or deployment with.
                    % Simulink Coder tools.
                    gensim(net);
                end
            end
            average_performance_error = sum(performance_error_per_iter) / iterate_count;
            ape_standard_deviation = std(performance_error_per_iter);
            average_performance = sum(performance_per_iter) / iterate_count;
            ap_standard_deviation = std(performance_per_iter);
            
            per_list(node_epoch_index, :) = [nodes(node_index) ...
                epoch_set(epoch_value_index) average_performance ...
                ap_standard_deviation average_performance_error ape_standard_deviation];
            node_epoch_index = node_epoch_index + 1;
        end
    end
    %%
    % plot the graph of error rate against epochs for each node values
    node2epochs = per_list(1:5, 2);
    node8epochs = per_list(6:10, 2);
    node32epochs = per_list(11:15, 2);

    node2errorrate = per_list(1:5, 3);
    node8errorrate = per_list(6:10, 3);
    node32errorrate = per_list(11:15, 3);

    error_graph_node_epoch = figure;
    plot(node2epochs, node2errorrate, 'r', node8epochs, node8errorrate, 'g',...
        node32epochs, node32errorrate, 'b'), legend('node: 2', 'node: 8', ...
        'node: 32'), xlabel('Epoch'), ylabel('Error');
    %% plot ends
    
    %%
    % finding the node and epoch value with lowest error rate
    sorted_performance_error_list = sortrows(per_list, [3 4])
    node = sorted_performance_error_list(1, 1);
    epoch = sorted_performance_error_list(1, 2); 
    %%
end
