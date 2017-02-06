clear; close all; clc
lambda=zeros(10,1);
for i=1:10
    lambda(i)=1/(10^(i-1));
end
err=zeros(5,10);
train_path={'..\..\data\spambase-cross-validation\Fold1\cv-train.txt',
    '..\..\data\spambase-cross-validation\Fold2\cv-train.txt',
    '..\..\data\spambase-cross-validation\Fold3\cv-train.txt',
    '..\..\data\spambase-cross-validation\Fold4\cv-train.txt',
    '..\..\data\spambase-cross-validation\Fold5\cv-train.txt'};

test_path={'..\..\data\spambase-cross-validation\Fold1\cv-test.txt',
    '..\..\data\spambase-cross-validation\Fold2\cv-test.txt',
    '..\..\data\spambase-cross-validation\Fold3\cv-test.txt',
    '..\..\data\spambase-cross-validation\Fold4\cv-test.txt',
    '..\..\data\spambase-cross-validation\Fold5\cv-test.txt'};

for j=1:5
    train = load(train_path{j});
    [m,n]=size(train);
    train(1,4)=0.1;
    train = (train - repmat(min(train),m,1)) ./ ( repmat(max(train),m,1) - repmat(min(train),m,1) );
    test = load(test_path{j});
    [m1,n1]=size(test);
    test = (test - repmat(min(test),m1,1)) ./ ( repmat(max(test),m1,1) - repmat(min(test),m1,1) );

    for i=1:10
        err(j,i) = Logistic_regularised(train,test,lambda(i));
    end
end
avg_err = sum(err)/5.0;
plot(lambda,avg_err);hold on;