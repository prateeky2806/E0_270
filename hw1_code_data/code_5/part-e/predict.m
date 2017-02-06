function [ Acc ] = predict( W,X_test, Y_test )
%PREDICT Summary of this function goes here
%   Detailed explanation goes here
size(X_test);
size(W);
H=X_test*W';
Y_pred = sign(H);
Acc = (length(find(Y_pred == Y_test))*100) / length(Y_test);
%confusion = confusionmat(Y_test, Y_pred);
end

