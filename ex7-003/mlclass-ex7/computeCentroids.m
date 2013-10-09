function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% ====================== NAIVE IMPLEMENTATION ===============
## centroids = zeros(K, n);
## for j = 1:K
##     s = zeros(1, n);
##     numJ = 0;
##     for i = 1:size(X, 1)
##         if (idx(i) == j)
##            numJ += 1;
##            s += X(i, :);
##         end
##     end
##     centroids(j, :) = s ./ numJ;
## end

% ===================== MORE EFFICIENT IMPLEMENTATION ======

centroids = zeros(K, n);
cCount = zeros(K, 1);
for i = 1:m
   centroids(idx(i), :) += X(i, :);
   cCount(idx(i), 1)++;
end

for j = 1:K
    if (cCount(j, 1) != 0)
       centroids(j, :) = centroids(j, :) / cCount(j, 1);
    else
        % set centroid randomly to new position
        data_min = min(X);
        data_max = max(X);
        data_diff = data_max .- data_min ;
        centroids(j, :) = (rand(1, n) .* data_diff) + data_min;
    end
end

% =============================================================

end
