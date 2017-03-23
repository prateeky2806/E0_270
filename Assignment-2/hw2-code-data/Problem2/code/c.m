clc;clear all;
data=load('../data/regression_dataset.mat');

train=data.train;
y_train=data.train_y;
test=data.test;
y_test=(data.test_y)';
weights=LLSR(train,y_train);
predTrain=[train ones(1000,1)]*weights;
predTest=[test ones(1000,1)]*weights;
disp(squared_error(predTrain,y_train))
disp(squared_error(predTest,y_test))