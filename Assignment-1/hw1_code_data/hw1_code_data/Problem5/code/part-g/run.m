clear ; close all; clc;
train_s = load('..\..\data\train_small.csv');
train_m = load('..\..\data\train_medium.csv');
train_l = load('..\..\data\train_large.csv');
test = load('..\..\data\test.csv');
X_train_s = train_s(:,1:(end-1)); Y_train_s = train_s(:,end);
[ms,ns]=size(X_train_s);
X_train_s = [ones(ms,1) X_train_s];

X_train_m = train_m(:,1:(end-1)); Y_train_m = train_m(:,end);
[mm,nm]=size(X_train_m);
X_train_m = [ones(mm,1) X_train_m];

X_train_l = train_l(:,1:(end-1)); Y_train_l = train_l(:,end);
[ml,nl]=size(X_train_l);
X_train_l = [ones(ml,1) X_train_l];

X_test = test(:,1:(end-1)); Y_test = test(:,end);
[m1,n1]=size(X_test);
X_test = [ones(m1,1) X_test];

time=zeros(3,1);
num_exp=[ms;mm;ml];
[weight_s, time(1)] = perceptron25(X_train_s, Y_train_s);
[weight_m, time(2)] = perceptron25(X_train_m, Y_train_m);
[weight_l, time(3)] = perceptron25(X_train_l, Y_train_l);
plot(num_exp,time);hold on;

Acc_train_s=predict(weight_s,X_test,Y_test)
Acc_train_m=predict(weight_m,X_test,Y_test)
Acc_train_l=predict(weight_l,X_test,Y_test)
