function [cost, gradient] = costfunction(theta, X, y)
[m,n]=size(X);
cost=0;
gradient = zeros(size(theta));
cost = (1/m)*sum(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta)));
gradient = (1/m)*(X'*(sigmoid(X*theta)-y));
end

