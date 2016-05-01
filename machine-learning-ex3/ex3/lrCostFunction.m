function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
h = sigmoid(X*theta);
f = ones(size(theta,1),1);
f(1) = 0;

% You need to return the following variables correctly 
J = 1/m * ((-y)' * log(h) - (1-y)' * log(1 - h));
grad = 1/m * X' * (sigmoid(X * theta) - y);

% Regularize
J = J + f' * (lambda/(2*m) * (theta .^ 2));
grad = grad + f .* ((lambda/m) * theta);

end
