function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2); % number of features
h = zeros(m,1);

J = 0;
grad = zeros(n,1);

for i = 1:m
    
    d = X(i,:) * theta;
    h(i) = 1.0/(1.0 + exp(-d));

end    



J = (-1.0/m) * sum(y .* log(h) + (1-y) .* log(1-h));

for j = 1:n    
    grad(j) = (1.0/m )* sum((h-y)' * X(:,j));
end    


% =============================================================

end
