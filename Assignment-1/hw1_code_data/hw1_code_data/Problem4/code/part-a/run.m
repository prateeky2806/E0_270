clear; close all; clc
train_full = load('train.txt');
[m,n]=size(train_full);
test = load('test.txt');
[m1,n1]=size(test);

err_train=zeros(10,1);
err_test=zeros(10,1);

for j=1:10
	idx = randperm(round((j/10)*m));
    train = train_full(1:round((j/10)*m), :);
    [a,b] = size(train);
    idx = randperm(round(a/3));
    train_test = train(idx,:);
    train_test = train;
    [err_train(j), err_test(j)] = Logistic_reg(train,train_test, test);
    fprintf('%4.2f\n',j);
end

num_exp=zeros(10,1);
for i=1:10
    num_exp(i)=round((i/10)*m);
end
plot(num_exp,err_train);hold on;
plot(num_exp,err_test); hold on;
