clear ; close all; clc;
train = load('..\..\data\train_small.csv');
%  [h,j]=size(train);
%  train = (train - repmat(min(train),h,1)) ./ ( repmat(max(train),h,1) - repmat(min(train),h,1) );
%train=zscore(train);
test = load('..\..\data\test.csv');
%  [h1,j1]=size(test);
%  test = (test - repmat(min(test),h1,1)) ./ ( repmat(max(test),h1,1) - repmat(min(test),h1,1) );
%test=zscore(test);
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
eta=0.55;
final_weight = winnow25(X_train, Y_train, eta);
Acc_train_test=zeros(25,1);
Acc_test = zeros(25,1);
for i=1:25
   Acc_train_test(i)=predict(final_weight(i,:),X_train,Y_train);
   Acc_test(i)=predict(final_weight(i,:),X_test,Y_test);
   %disp(i)
end
q=zeros(25,1);
for i=1:25
    q(i)=i;
end
scatter(q,Acc_test)
scatter(q,Acc_train_test)