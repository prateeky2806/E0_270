function [ Acc ] = predict( W,X_test, Y_test )
%PREDICT Summary of this function goes here
%   Detailed explanation goes here
[m,n]=size(X_test);
size(W);
H=X_test*W';
Y1 = (H>=0.5/n)
Y0 = (H<0.5/n);
Y_pred=zeros(size(H));
Y_pred(Y1)=1;
Y_pred(Y0)=-1;
Y_pred;
Acc = (length(find(Y_pred == Y_test))*100) / length(Y_test);
%confusion = confusionmat(Y_test, Y_pred);
end

