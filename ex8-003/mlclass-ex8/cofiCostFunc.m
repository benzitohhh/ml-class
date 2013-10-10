function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%

%R:     m x u
%Y:     m x u
%Theta: u x n
%X:     m x n

% COST
J = (Theta * X')';  % m x u
J = J - Y;
J = J .^ 2;
J = J .* R;
J = 0.5 * sum(sum(J));       % 1 x 1


% X_grad
for i = 1:num_movies
    idx = find(R(i, :) == 1);    %% 1  x u'   indices of users that have rated movie i
    Theta_temp = Theta(idx, :);  %% u' x n    feature vectors for users that have rated movie i
    Y_temp = Y(i, idx);          %% 1  x u'   ratings for movie i from users that have rated it
    g = X(i, :) * Theta_temp';   %% 1  x u'
    g = g - Y_temp;
    g = g * Theta_temp;          %% 1  x n
    X_grad(i, :) = g;
end

% Theta_grad
for j = 1:num_users
    idx = find(R(:, j) == 1);    %% m' x 1   indices of movies that have been rated by user j
    X_temp = X(idx, :);          %% m' x n   feature vectors for movies that have been rated by user j
    Y_temp = Y(idx, j);          %% m' x 1   ratings for movies that have been rated by user j
    g = X_temp * Theta(j, :)';   %% m' x 1
    g = g - Y_temp;
    g = g';                      %% 1  x m'
    g = g * X_temp;              %% 1  x n
    Theta_grad(j, :) = g;
end














% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
