function [ final_weight , updates] = perceptron( X_train, Y_train, eta )
%PERCEPTRON Summary of this function goes here
%   Detailed explanation goes here

[m,n] = size(X_train);
weights = zeros(n,1);
X=zeros(25,n);
updates=0;
for j=1:25
    for i=1:m
        if Y_train(i)~= sign(X_train(i,:)*(weights))%weights is a n*1 matrix
            weights = weights +(eta*(Y_train(i)*(X_train(i,:))'));
            updates = updates + 1;
        end
    end
    X(j,:) = weights' ;
end
final_weight = (X(25,:));

