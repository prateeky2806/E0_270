clear ; close all; clc;
train = load('..\..\data\train_small.csv');
test = load('..\..\data\test.csv');
X_train = train(:,1:(end-1)); Y_train = train(:,end);
[m,n]=size(X_train);
X_train = [ones(m,1) X_train];
X_test = test(:,1:(end-1)); Y_test = test(:,end);
[m1,n1]=size(X_test);
X_test = [ones(m1,1) X_test];
train_test = datasample(train,round((33/100)*m),'Replace',false);
X_train_test = train_test(:,1:(end-1));
Y_train_test = train_test(:,end);
[m2,n2]=size(X_train_test);
X_train_test = [ones(m2,1) X_train_test];
final_weight = perceptron25(X_train, Y_train);
Acc_train_test=zeros(25,1);
Acc_test = zeros(25,1);
for i=1:25
   Acc_train_test(i)=predict(final_weight(i,:),X_train_test,Y_train_test);
   Acc_test(i)=predict(final_weight(i,:),X_test,Y_test);
end
q=zeros(25,1);
for i=1:25
    q(i)=i;
end
%scatter(q,Acc_test)
max10 = final_weight(25,:);
[sortedValues,sortIndex] = sort(max10,'descend');
fprintf('Indices of the highest value of weight, if index is "x"\n' );
fprintf('It basically means final_weights(25,x)\n');
fprintf('final_weights(25,:) dentoes the final weights after all 25 iterations');
fprintf('"x-1" word in vocabulary is most occured as we appended ones while training');
maxIndex = sortIndex(1:10);
index=maxIndex-1
% data=fopen('..\..\data\imdb_vocab.csv','r');
% formatSpec = '%s%[^\n\r]';
%vocab = vocab_import('..\..\data\imdb_vocab.csv',1,14666);
%words=vocab(index)
