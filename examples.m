v = zeros(10,1)
for i=1:10,
  v(i) = 2^i;
end

i = 1;
while i <=5,
  v(i) = 100;
  i = i +1;
end

i = 1;
while true,
  v(i) = 999;
  i = i + 1;
  if i == 6,
    break;
  end;
end;

a = 4
if a == 1,
  disp('yay');
elseif a == 2,
  disp('bla');
else
  disp('arghhh');
end

function y = squareThisNumber(x)
  y = x^2;
end
# also you can save this in a file called squareThisNumber.m
# make sure it is on the "path"
# i.e. pwd
# i.e. addpath('...')

function [y1, y2] = yup2(x)
  y1 = x^3;
  y2 = x^2 + 1;
end
# access like this: [y1 y2] = yup2(2)

function J = costFunctionJ(X, y, theta)
  % X is the "design matrix" containing our training examples
  % y is the class labels

  m = size(X, 1);  # num rows in X
  predictions = X * theta;
  sqrErrors = (predictions -y) .^ 2;
  J = 1 / (2 * m) * sum(sqrErrors);
end
