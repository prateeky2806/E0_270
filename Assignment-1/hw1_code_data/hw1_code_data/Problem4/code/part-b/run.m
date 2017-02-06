clear; close all; clc
train = load('..\..\data\train.txt');
[m,n]=size(train);
train(1,4)=0.1;
train = (train - repmat(min(train),m,1)) ./ ( repmat(max(train),m,1) - repmat(min(train),m,1) );
test = load('..\..\data\test.txt');
[m1,n1]=size(test);
test = (test - repmat(min(test),m1,1)) ./ ( repmat(max(test),m1,1) - repmat(min(test),m1,1) );
train_test=datasample(train,round(m/3),'Replace',false);
[m2,n2]=size(train_test);
lambda=zeros(10,1);
for i=1:10
    lambda(i)=1/(10^(i-1));
end
test_err=zeros(10,1);
train_err=zeros(10,1);
% Returns test error for all values of lambda ranging fron {1,10^(-1), ..., 10^(-10)}

% train_err1 = Logistic_regularised(train,train_test,lambda(1));
% test_err1=Logistic_regularised(train, test, lambda(1));

for j=1:10
    train_err(j) = Logistic_regularised(train,train_test,lambda(j));
    test_err(j)=Logistic_regularised(train, test, lambda(j));
    disp(j)
end
plot(lambda, test_err,'Marker','o','MarkerFaceColor','c',...
    'Color','b');hold on;
plot(lambda, train_err,'Marker','o','MarkerFaceColor','y',...
    'Color','g');hold on;
xlabel('lambda');
ylabel('Error(fraction)');
legend('Test Error','Training Error');
title('Error vs. lambda');
