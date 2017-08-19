function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 



m = length(y); % number of training examples
n = size(X,2); % number of features
h = zeros(m,1);

J = 0;
grad = zeros(n,1);


for i = 1:m
    
    d = X(i,:) * theta;
    h(i) = 1.0/(1.0 + exp(-d));
  
end    


t1 = theta(2:n,:);
J = (-1.0/m) * sum(y .* log(h) + (1-y) .* log(1-h)) + (lambda/(2.0 * m)) * t1' * t1;

for j = 1:n    
    if (j == 1)
      grad(j) = (1.0/m )* sum((h-y)' * X(:,j));
    else 
      grad(j) = (1.0/m )* sum((h-y)' * X(:,j)) + (lambda/m)* theta(j) ;
    end
    
end    



% =============================================================

end
