clear ; close all; clc;
train = load('..\..\data\train_small.csv');
test = load('..\..\data\test.csv');
[m,n]=size(train);
X_test = test(:,1:(end-1)); Y_test = test(:,end);
train_test = datasample(train,round((33/100)*m),'Replace',false);
X_train_test = train_test(:,1:(end-1));
Y_train_test = train_test(:,end);
[W,b] = naiveBayes(train);
[Acc_train, confusion_train] = predict(W,b,X_train_test, Y_train_test);
[Acc_test, confusion_test] = predict(W,b,X_test, Y_test);
fprintf('Training accuracy = %4.2f\n', Acc_train)
fprintf('Testing accuracy = %4.2f\n', Acc_test)
fprintf('Test Confusion matrix is: \n')
disp(confusion_test);
