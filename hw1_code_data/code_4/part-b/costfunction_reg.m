function [cost, gradient] = costfunction_reg(theta, X,y, lambda)
m=length(y);
cost=0;
%gradient = zeros(size(theta));
H = sigmoid(X*theta);
T = sum(y'*log(H) + (1 - y)'*log(1 - H));
cost = (-1/m)*T + (lambda/(2*m))*sum(theta(2:end).^2);

gradient = zeros(size(theta));
gradient(1)=1/m*(sum((X(1,:)*H(1)-(X(1,:))*y(1))));
k=length(theta);
 
for j=2:k
    gradient(j)=(1/m)*(sum((X(j,:))'*H(j)-(X(j,:))'*y(j))+lambda.*theta(j,1));
end

