function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
X = X';               %% 400 x 5000
X = [ones(1, m); X];  %% 401 x 5000
z2 = Theta1 * X;
a2 = sigmoid(z2);     %% 25 x 5000

a2 = [ones(1, m); a2];%% 26 x 5000
z3 = Theta2 * a2;     %% 10 x 5000
a3 = sigmoid(z3);      %%

% Currently, y has a single val per example (int labels between 0 and 10 inclusive).
% i.e. example 1 might be '3'
% i.e. example 2 might be '4'
% We need to transform y to a matrix, with a K-dimensional vector per example.
% i.e. example 1 might become [0 0 1 0 0 0 0 0 0 0], denoting it is in class 3
% i.e. example 2 might become [0 0 0 1 0 0 0 0 0 0], denoting it is in class 4
% y: 5000 x 1
temp = [];
for i = 1:num_labels
    temp = [temp, y==i];
endfor                %% 5000 x 10
y = temp';            %% 10 x 5000

costY1 = y .* -log(a3);                    % 10 x 5000
costY0 = (1 - y) .* -log(1 - a3);          % 10 x 5000
cost = costY0 + costY1;                    % 10 x 5000
cost = 1 / m * sum(sum(cost));             % 1 x 1

reg1 = sum(sum(Theta1(:, 2:end) .^ 2));    % exclude theta(1) from regularization
reg2 = sum(sum(Theta2(:, 2:end) .^ 2));    % exclude theta(1) from regularization
reg = reg1 + reg2;
reg = 1 / (2*m) * lambda * reg;            % 1 x 1

J = cost + reg;                            % 1 x 1


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

%X;       % 401 x 5000
%y;       % 10 x 5000
X = X';   % 5000 x 401
y = y';   % 5000 x 10
%Theta1   % 25 x 401
%Theta2   % 10 x 26

for t = 1:m
    a1 = X(t,:)';     % 401 x 1
    z2 = Theta1 * a1; % 25 x 1
    a2 = sigmoid(z2);

    a2 = [1; a2];     % 26 x 1
    z3 = Theta2 * a2; % 10 x 1
    a3 = sigmoid(z3);
endfor

## grad = (h - y)' * X;
## grad = 1 / m * grad;
## grad = grad';



%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%





















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
