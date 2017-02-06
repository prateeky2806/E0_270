clear ; close all; clc;
train = load('..\..\data\train_small.csv');
% [m,n]=size(X_train);
test = load('..\..\data\test.csv');
updates=zeros(20,1);
weights=zeros(20,14667);
for i=1:20
    train = shuffle(train);
    X_train = train(:,1:(end-1)); Y_train = train(:,end);
    [m,n]=size(X_train);
    X_train = [ones(m,1) X_train];
    [weights(i,:),updates(i)] = perceptron25(X_train, Y_train);
end
histogram(updates,8); hold on;
xlabel('No. of Updates');
ylabel('count');
title('Histogram plot, No. of updates vs count of range');

R=max(max(X_train));
W=weights(10,:)*weights(10,:)';
gamma=(R*W)/((520)^(0.5))
