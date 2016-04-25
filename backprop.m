%{
Neural network backpropagation

This code implements the basic backpropagation of error learning algorithm.
The network has hidden neurons and a linear output neuron.

Args:
input: Matrix where each row is a feature vector/observation, making the
    columns represent each feature.
expected_output: Nx1 vector where each element is the classification of
    the corresponding input in the input matrix. N is the number of feature
    vectors in input.
hidden_neurons: Number of hidden neurons to use.
max_iterations: Maximum number of iterations to train the network.
learning_rate: Relative impact the error has on adjusting the weights.
max_error: Threshold where we can exit early if the total error is less
    than this.

Returns:
weight_input_hidden: LxM matrix of weight vectors for each hidden neuron
    where L is the number of features (+1 bias) and M is the number of
    hidden_neurons.
weight_hidden_output: 1xM matrix of weight vectors for each hidden_neuron.
    1 because the expected output is meant to be 1 dimensional
    (the classification).
pred: Predictions for the classification of the input as a result of
    the weights. This has the same dimensions as the expected_output.
err: 1D vector of the total errors accumulated from training.
%}
function [weight_input_hidden, weight_hidden_output, pred, err] = ...
    backprop(input, expected_output, hidden_neurons, max_iterations,...
    learning_rate, max_error)
    lr = learning_rate;  % learning rate
    err_max = max_error;

    % ------- load in the data -------
    % XOR data
    train_inp = input;  % Each row is a feature vector
    train_out = expected_output;  % Each row is the classificatin

    % check same number of patterns in each
    if size(train_inp,1) ~= size(train_out, 1)
       disp('ERROR: data mismatch')
       return 
    end

    % read how many patterns/observations
    patterns = size(train_inp, 1);

    % add a bias as an input
    bias = ones(patterns, 1);
    train_inp = [train_inp bias];  % +1 feature(the bias) moves us up 1 dimension

    % read how many inputs/dimensions/features (including 1 bias)
    inputs = size(train_inp, 2);

    %---------- set weights -----------------
    % set initial random weights
    weight_input_hidden = randn(inputs, hidden_neurons);
    weight_hidden_output = randn(1, hidden_neurons);

    %--- Learning Starts Here ---------
    for iter = 1:max_iterations
        %loop through the patterns, selecting randomly
        for patnum = 1:patterns
            % set the current pattern
            this_pat = train_inp(patnum, :);
            act = train_out(patnum);  % actual/expected

            % calculate the current error for this pattern
            hval = tanh(this_pat * weight_input_hidden)';
            pred = hval' * weight_hidden_output';
            error = pred - act;

            % adjust weight hidden - output
            delta_HO = error .* lr .* hval;
            weight_hidden_output = weight_hidden_output - delta_HO';

            % adjust the weights input - hidden
            delta_IH= lr .* error .* weight_hidden_output' .* (1-hval) * this_pat;
            weight_input_hidden = weight_input_hidden - delta_IH';
        end
        % -- another iteration finished

        % Get overall network error at end of each iteration
        pred = weight_hidden_output * tanh(train_inp * weight_input_hidden)';
        error = pred' - train_out;
        err(iter) = (sum(error.^2))^0.5;

        % stop if error is small
        if err(iter) < err_max
            fprintf('converged at iteration: %d\n',iter);
            break 
        end
    end
    pred = pred';