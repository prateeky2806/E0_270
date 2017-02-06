function [ X ] = winnow25( X_train, Y_train, eta )
%PERCEPTRON Summary of this function goes here
%   Detailed explanation goes here

[m,n] = size(X_train);
weights = -ones(n,1)./n
X=ones(25,n);
updates=0;
for j=1:25
    for i=1:m
        if Y_train(i)~= sign(X_train(i,:)*(weights))%weights is a n*1 matrix
            for k = 1:n
                power = eta * Y_train(i)*(X_train(i,k));
                weights(k) = weights(k)*exp(power);
            end
            weights = weights./sum(weights);
            updates = updates + 1;
        end
    end
    X(j,:) = weights' ;
end

