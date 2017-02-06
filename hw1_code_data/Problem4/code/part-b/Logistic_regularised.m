function [test_err] = Logistic_regularised(train, test, lambda)
X_train = train(:,1:(end-1)); Y_train = train(:,end);
X_test = test(:,1:(end-1)); Y_test = test(:,end);
[m,n] = size(X_train);
[m1,n1] = size(X_test);
X_train = [ones(m,1) X_train];
[m2,n2]=size(X_train);
X_test = [ones(m1,1) X_test];
theta = zeros(n2, 1);
%options = optimset('GradObj','on');
[final_theta, cost] = fminunc(@(t)(costfunction_reg(t,X_train,Y_train,lambda)), theta);
Y_pred = predict(final_theta, X_test);
training_acc = mean(double(Y_pred==Y_test))*100;
%fprintf('Train Accuracy: %f\n', training_acc);
test_err = classification_error(Y_pred, Y_test);
fprintf('Test error: %f\n', test_err);
end
