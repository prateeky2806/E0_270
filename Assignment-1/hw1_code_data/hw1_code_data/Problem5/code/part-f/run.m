clear ; close all; clc;
train = load('..\..\data\train_small.csv');
test = load('..\..\data\test.csv');
X_train = train(:,1:(end-1)); Y_train = train(:,end);
[m,n]=size(X_train);
X_train = [ones(m,1) X_train];
X_test = test(:,1:(end-1)); Y_test = test(:,end);
[m1,n1]=size(X_test);
X_test = [ones(m1,1) X_test];
% train_test = datasample(train,round((33/100)*m),'Replace',false);
% X_train_test = train_test(:,1:(end-1));
% Y_train_test = train_test(:,end);
[m2,n2]=size(X_train);
% X_train_test = [ones(m2,1) X_train_test];
num=12;
eta=zeros(num,1);
eta(1)=0.05;
for i=2:num
    eta(i)=0.05+eta(i-1);
end
weights=zeros(num,n+1);
updates=zeros(num,1);
Acc_train_test=zeros(num,1);
Acc_test = zeros(num,1);
for i=1:num
   [weight, update] = perceptron25(X_train, Y_train,eta(i));
   Acc_train_test(i)=predict(weight,X_train,Y_train);
   Acc_test(i)=predict(weight,X_test,Y_test);
   size(weight);
   weights(i,:)=weight;
   updates(i)=update;
end
plot(eta,updates);
T=table(eta,updates,Acc_train_test,Acc_test);
%fig = uifigure;
%tab_fig = uitable(fig,);

