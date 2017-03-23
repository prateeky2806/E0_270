function [  ] = func_b ( kerneltype )

fid = fopen(strcat(kerneltype,'.txt'),'w');

data=importdata('..\data\regression_folds.mat');
err=zeros(1,5);
avg_err=zeros(1,3);
k=1;
cval=[0.01,1,100];
for C = cval
    for i=1:5
        %training
        x=data.fold_train(i,:,:);
        y=data.fold_train_y(i,:);
        SVR_learner( reshape(x,size(x,2),size(x,3)) , y' , C , kerneltype , 3 , 0.1 );
        
        %testing
        x=data.fold_test(i,:,:);
        truth=data.fold_test_y(i,:);
        predicted=SVR_regressor(reshape(x,size(x,2),size(x,3)));
        err(i)=squared_error(predicted,truth');
        fprintf(fid,'C =%f ,fold= %d , we get error=%f\n',C,i,err(i));
    end
    fprintf(fid,'for C=%f we have average error of %f\n\n\n',C,mean(err));
    avg_err(k)=mean(err);
    k=k+1;
end

[val,pos]=min(avg_err);
C=cval(pos);
fprintf(fid,'C= %f gives trhe lowest cross valid. average error = %f\n',C,avg_err(pos));

dt=importdata('..\data\regression_dataset.mat');

cval=[0.01,1,100];
for C=cval
kerneltype='poly';%poly
SVR_learner(dt.train,dt.train_y,C,kerneltype,3,0.1);

predicted=SVR_regressor(dt.train);
err=squared_error(predicted,dt.train_y);
fprintf('the training error is %f \n',err);

predicted=SVR_regressor(dt.test);
err=squared_error(predicted,dt.test_y');
fprintf('the test error is %f \n',err);
end
end