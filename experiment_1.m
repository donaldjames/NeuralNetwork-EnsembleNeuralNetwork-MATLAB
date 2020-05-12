% Experiment 1
% setting the nodes [2 8 32] and epochs [4 8 16 32 64] and running the
% 15 combinations 'iterate_count' of times. The average error rate,
% performance mean and standard deviation is calculated and stored into a
% multidimensional array for the ease of calculations
function [node, epoch, per_list] = experiment_1(iterate_count)
    load cancer_dataset;
    trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

    % Create a Pattern Recognition Network
    nodes = [2 8 32];
    epoch_set = [4 8 16 32 64];
    per_list = zeros(15, 6);
    node_epoch_index = 1;
    for node_index = 1:length(nodes)
        hiddenLayerSize = nodes(node_index);
        net = patternnet(hiddenLayerSize, trainFcn);
        for epoch_value_index = 1:length(epoch_set)
            test_performance = zeros(iterate_count, 1);
            train_performance = zeros(iterate_count, 1);
            net.trainParam.epochs = epoch_set(epoch_value_index);
            for i = 1:iterate_count
                x = cancerInputs;
                t = cancerTargets;
                
                net.input.processFcns = {'removeconstantrows','mapminmax'};
                net.divideFcn = 'dividerand';  % Divide data randomly
                net.divideMode = 'sample';  % Divide up every sample
                net.performFcn = 'mae';  % Mean absolute error

                net.divideParam.trainRatio = 50/100;
                net.divideParam.valRatio = 0/100;
                net.divideParam.testRatio = 50/100;

                % calling the training function
                [net, tr] = train(net,x,t);
                y = net(x);
                e = gsubtract(t,y);
                performance = perform(net,t,y);
                tind = vec2ind(t);
                yind = vec2ind(y);
                percentErrors = sum(tind ~= yind)/numel(tind);

                % Recalculate Training, Validation and Test Performance
                trainTargets = t .* tr.trainMask{1};
                testTargets = t .* tr.testMask{1};
                trainPerformance = perform(net,trainTargets,y)
                testPerformance = perform(net,testTargets,y)

                % Storing the performance of each run to performance_list
                test_performance(i, :) = testPerformance;
                train_performance(i, :) = trainPerformance;
            end
            average_train_performance = sum(train_performance) / iterate_count;
            train_per_standard_deviation = std(train_performance);
            average_test_performance = sum(test_performance) / iterate_count;
            test_per_standard_deviation = std(test_performance);
            
            per_list(node_epoch_index, :) = [nodes(node_index) ...
                epoch_set(epoch_value_index) average_test_performance ...
                test_per_standard_deviation average_train_performance train_per_standard_deviation];
            node_epoch_index = node_epoch_index + 1;
        end
    end
    % plot the graph of error rate against epochs for each node values
    node2epochs = per_list(1:5, 2);
    node8epochs = per_list(6:10, 2);
    node32epochs = per_list(11:15, 2);

    node2testerrorrate = per_list(1:5, 3);
    node8testerrorrate = per_list(6:10, 3);
    node32testerrorrate = per_list(11:15, 3);

    figure;
    plot(node2epochs, node2testerrorrate, 'r', node8epochs, node8testerrorrate, 'g',...
        node32epochs, node32testerrorrate, 'b'), legend('node: 2', 'node: 8', 'node: 32'), ...
        xlabel('Epoch'), ylabel('Error'), title('Test Error vs Epoch');
    %% plot ends
    
    %%
    % finding the node and epoch value with lowest error rate
    sorted_performance_error_list = sortrows(per_list, [3 4])
    node = sorted_performance_error_list(1, 1);
    epoch = sorted_performance_error_list(1, 2); 
    %%
end
