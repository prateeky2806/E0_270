function [ weights ] = perceptron( X_train, Y_train, weight_initial )
%PERCEPTRON Summary of this function goes here
%   Detailed explanation goes here

[m,n] = size(X_train);
weights = weight_initial;
total_updates=0;
while run==1
    for i=1:m
        updates = 0;
        if Y_train(i)!= sign(X_train(i,:)*(weights'))
            weights = weights +(Y_train(i)*X_train(i,:));
            updates = updates + 1;
        end
        total_updates = Total_updates + updates;
        if updates > 0
            run=1;
        else
            run=0;
        end
    end
end

