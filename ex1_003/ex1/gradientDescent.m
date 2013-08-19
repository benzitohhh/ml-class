function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    #disp(sprintf('iter: %d', iter));
    h = X * theta;
    #disp('h:'), disp(h);

    delta = (h - y)' * X;
    delta = 1 / m * delta;
    delta = delta';
    #disp('delta:'), disp(delta);

    theta = theta - (alpha * delta);
    #disp('theta:'), disp(theta);
    % ============================================================

    % Save the cost J in every iteration
    cost = computeCost(X, y, theta);
    #disp(sprintf('cost(iter=%d): %0.2f', i, cost));
    J_history(iter) = cost;


end

end
