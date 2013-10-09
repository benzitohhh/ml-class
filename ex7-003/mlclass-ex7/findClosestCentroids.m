function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% ===================== NAIVE IMPLEMENTATION ===============
m = size(X, 1);
## idx = zeros(size(X,1), 1);
## for i = 1:m
##     dists = sum((centroids .- X(i, :)), 2);
##     [dummy, idx(i)] = min(dists);
## end


% ===================== MORE EFFICIENT IMPLEMENTATION ======
## The core of this problem is to compute a distance matrix D of size m x 3
## that contains the pairwise distances between all data points in X and
## all data points in C. The Euclidean distance between the i-th vector x_i
## in X and the j-th vector c_j in C can be rewritten as:

## |x_i-c_j|^2 = |x_i|^2 - 2<x_i, c_j> + |c_j|^2

## where <,> refers to inner product. The right-hand side of this equation
## can be easily vectorized, because the inner product of all pairs is just
## X * C' which is BLAS3 operation. This way of computing the distance
## matrix is known as dist2 function in the book Pattern Recognition and
## Machine Learning by Christopher Bishop. I copy the function below with a
## little modification.

tempx = full(sum(X.^2, 2));             % m x 1
tempc = full(sum(centroids.^2, 2).');   % 1 x k
D = -2*(X * centroids.');               % m x k
D = bsxfun(@plus, D, tempx);
D = bsxfun(@plus, D, tempc);

## The full here is used in case X or C is a sparse matrix.

## Note: The distance matrix D computed this way might have tiny negative
## entries due to numerical rounding error. To guard against this case, use
## D = max(D, 0);

## Now, the indices of the closest vector in C can be retrieved from D:

[~, idx] = min(D, [], 2);
% =============================================================

end
