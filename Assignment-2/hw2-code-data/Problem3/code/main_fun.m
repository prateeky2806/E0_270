function [  ] = main_fun()% train,test )
dt=importdata('..\data\regression_dataset.mat');
fid = fopen('exp.txt','w');
SVR_learner(dt.train(1:10,:),dt.train_y(1:10,:),100,'rbf',3,0.1);
predicted=SVR_regressor(dt.test);
fprintf(fid,'squarred error is %f',squared_error(predicted,dt.test_y'));
end