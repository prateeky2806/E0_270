function [P] = predict(theta, X)
[m,n] = size(X);
%P=zeros(m,1);
P = (sigmoid(X*theta) >= 0.5);
Q = (sigmoid(X*theta) < 0.5);
H=zeros(m,1);
H(find(P))=1;
H(find(Q))=-1;
end

