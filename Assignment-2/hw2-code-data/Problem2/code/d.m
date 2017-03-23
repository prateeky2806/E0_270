foldSet=load('../data/regression_folds');
dataSet=load('../data/regression_dataset');
L=[0.01 0.1 1 10 100];
err=[];
fold_test    = foldSet.fold_test;
fold_train   = foldSet.fold_train;
y_fold_test  = foldSet.fold_test_y;
y_fold_train = foldSet.fold_train_y;
train=dataSet.train;
y_train=dataSet.train_y;
test=dataSet.test;
test_y=dataSet.test_y;

weights=LRR(train , y_train ,  0.1);
 Pred=[];
     
     for k=1:1000
     Pred(end+1)=[train(k,:) 1]*weights;
     end
     squared_error(Pred,y_train')
     
for i=1:5
    
    lambda=L(i);
    temp=0;
    for j=1:5
     Test=squeeze(fold_test(j,:,:));
     Train=squeeze(fold_train(j,:,:));
     
     y_test=squeeze(y_fold_test(j,:));
     y_train=squeeze(y_fold_train(j,:));
     
     weights=LRR(Train , y_train' ,  lambda);
     pred=[];
     
     for k=1:200
     pred(end+1)=[Test(k,:) 1]*weights;
     end
     temp=temp+squared_error(pred,y_test);
    end
    err(end+1)=temp/5;
end
err;

[minlambda,index]=min(err);
