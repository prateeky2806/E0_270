clear; close all; clc
train_full = load('..\..\data\train.txt');
 [m,n]=size(train_full);
% train_full(1,4)=0.1;
% train_full = (train_full - repmat(min(train_full),m,1)) ./ ( repmat(max(train_full),m,1) - repmat(min(train_full),m,1) );
%train_full = zscore(train_full);
test = load('..\..\data\test.txt');
 [m1,n1]=size(test);
% test = (test - repmat(min(test),m1,1)) ./ ( repmat(max(test),m1,1) - repmat(min(test),m1,1) );
%test = zscore(test);

err_train=zeros(10,1);
err_test=zeros(10,1);
% train=datasample(train_full,round((3/10)*m),'Replace',false);
%     [a,b]=size(train);
%     %train_test is a subsample of training data which we will use for
%     %testing as mentioned in the question and we have a seperate test set too.
%     %train_test=zeros(round(a/3));
%     train_test=datasample(train,round(a/3),'Replace',false);
%     [err_train1, err_test1] = Logistic_reg(train,train_test, test);
 
for j=1:10
    %train=zeros(round((j/10)*m,1));
    train=datasample(train_full,round((j/10)*m),'Replace',false);
    [a,b]=size(train);
    %train_test is a subsample of training data which we will use for
    %testing as mentioned in the question and we have a seperate test set too.
    %train_test=zeros(round(a/3));
    train_test=datasample(train,round(a/3),'Replace',false);
    [err_train(j), err_test(j)] = Logistic_reg(train,train_test, test);
    fprintf('%4.2f',j);
end
num_exp=zeros(10,1);
for i=1:10
    num_exp(i)=round((i/10)*m);
end
plot(num_exp,err_train);hold on;
figure;plot(num_exp,err_test); hold on;
