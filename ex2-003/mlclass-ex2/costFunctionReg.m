function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = X * theta; % predictions
h = sigmoid(h); % apply sigmoid
costY1 = y .* -log(h);
costY0 = (1 - y) .* -log(1 - h);
cost = costY0 + costY1;
cost = 1 / m * sum(cost);

reg = sum(theta(2:end) .^ 2);    % exclude theta(1) from regularization
reg = 1 / (2*m) * lambda * reg;

J = cost + reg;

grad = (h - y)' * X;
grad = 1 / m * grad;
grad = grad';
grad(2:end) = grad(2:end) .+ (1/m * lambda * theta(2:end));

% =============================================================

end
