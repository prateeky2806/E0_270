function [err_train_test, err_test] = Logistic_reg(train, train_test, test)
%train_test is a subsample of training data which we will use for -
%testing as mentioned in the question and we have a seperate test set too.

X_train = train(:,1:(end-1));
Y_train = train(:,end);
Y_train = Y_train > 0;

X_test = test(:,1:(end-1));
Y_test = test(:,end);


[m,n] = size(X_train);
[m1,n1] = size(X_test);

X_train = [ones(m,1) X_train];
X_test = [ones(m1,1) X_test];

X_train_test = train_test(:,1:(end-1));
Y_train_test = train_test(:,end);

[m2,n2] = size(X_train_test);

X_train_test = [ones(m2,1) X_train_test];

theta = zeros(n+1, 1);
options = optimset('GradObj','on');
[final_theta, cost] = fminunc(@(k)(costfunction(k,X_train,Y_train)), theta, options);

Y_pred_train = predict(final_theta, X_train_test);

err_train_test = classification_error(predict(final_theta, X_train_test), Y_train_test);
err_test = classification_error(predict(final_theta, X_test), Y_test);

end
