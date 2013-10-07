function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% X:     m x d
% theta: d x 1

h = theta' * X';            % 1 x m
h = h';                     % m x 1
err = (h - y) .^ 2;         % m x 1
J = 1 / (2 * m) * sum(err);
reg = lambda / (2 * m) * sum(theta(2:end) .^ 2); % exclude bias term from regularization
J = J + reg;

grad = (h - y)' * X;                   % 1 x d
grad = grad';                          % d x 1
grad = 1 / m * grad;
reg = [0; lambda / m * theta(2:end)];  % d x 1 (exclude bias from regularaization)
grad = grad + reg;

% =========================================================================

grad = grad(:);

end
