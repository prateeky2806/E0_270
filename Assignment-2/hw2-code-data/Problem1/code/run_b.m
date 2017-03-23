clear all; close all; clc;
features = 1:2;
C_val = [1, 10, 100, 1000, 10000];
sigma_val = [1/32, 1/4, 1, 4, 32];
num_folds = 5;

% Loading Data
X = load('../data/Synth3/train.txt');
X_test = load('../data/Synth3/test.txt');
y = X(:, 3);
X = X(:, 1:2);
m = size(X, 1);
n = size(X, 2);
y_test = X_test(:, 3);
X_test = X_test(:, 1:2);
m_test = size(X_test, 1);
n_test = size(X_test, 2);

% linear kernel
kerneltype = 'linear';
r = 0;

c_sigma_err_train = zeros(length(C_val), 1);
c_err_val = zeros(length(C_val), 1);
k = 1;
for c = C_val
	train_err = zeros(num_folds, 1);
	val_err = zeros(num_folds, 1);
	for fold = 1:num_folds
		val_range = (fold-1)*(m/num_folds)+1 : fold*(m/num_folds);
		X_val = X(val_range, :);
		y_val = y(val_range, :);
		X_train = X(setdiff(1:m, val_range), :);
		y_train = y(setdiff(1:m, val_range), :);
			
		model = SVM_learner(X_train, y_train, c, kerneltype, r);
		train_err = classification_error(SVM_classifier(X_train, model), y_train);
		val_err = classification_error(SVM_classifier(X_val, model), y_val);
		train_err(fold) = train_err;
		val_err(fold) = val_err;
		
    end
	c_err_val(k) = mean(val_err);
	c_sigma_err_train(k) = mean(train_err);
	k = k + 1;
end
[~, index] = min(c_err_val);
c_min = C_val(index);

model = SVM_learner(X, y, c_min, kerneltype, r);

test_err = classification_error(SVM_classifier(X_test, model), y_test);
train_err = classification_error(SVM_classifier(X, model), y);
figure;
decision_boundary_SVM(X_test, y_test, model);

%{
% degree 2 Polynomial kernel
kerneltype = 'poly';
r = 2;

c_sigma_err_train = zeros(length(C_val), 1);
c_err_val = zeros(length(C_val), 1);
k = 1;
for c = C_val
	train_err = zeros(num_folds, 1);
	val_err = zeros(num_folds, 1);
	cv = cvpartition(size(X, 1), 'KFold', num_folds);
	for fold = 1:num_folds		
		% Prepare the data for training and validation
		X_train = X(cv.training(fold), :);
		y_train = y(cv.training(fold), :);
		X_val = X(cv.test(fold), :);
		y_val = y(cv.test(fold), :);
				
		% Train the model
		model = SVM_learner(X_train, y_train, c, kerneltype, r);
		train_err = classification_error(SVM_classifier(X_train, model), y_train);
		val_err = classification_error(SVM_classifier(X_val, model), y_val);
		
		% Record the data
		train_err(fold) = train_err;
		val_err(fold) = val_err;
    end
	c_err_val(k) = mean(val_err);
	c_sigma_err_train(k) = mean(train_err);
	k = k + 1;
end
[~, index] = min(c_err_val);
c_min = C_val(index);

model = SVM_learner(X, y, c_min, kerneltype, r);
test_err = classification_error(SVM_classifier(X_test, model), y_test);
train_err = classification_error(SVM_classifier(X, model), y);
figure
decision_boundary_SVM(X_test, y_test, model);
%}
%{
% degree 3 polynomial kernel
kerneltype = 'poly';
r = 3;

c_sigma_err_train = zeros(length(C_val), 1);
c_err_val = zeros(length(C_val), 1);
k = 1;
for c = C_val
	train_err = zeros(num_folds, 1);
	val_err = zeros(num_folds, 1);
	cv = cvpartition(size(X, 1), 'KFold', num_folds);
	for fold = 1:num_folds		
		X_train = X(cv.training(fold), :);
		y_train = y(cv.training(fold), :);
		X_val = X(cv.test(fold), :);
		y_val = y(cv.test(fold), :);
		model = SVM_learner(X_train, y_train, c, kerneltype, r);
		train_err = classification_error(SVM_classifier(X_train, model), y_train);
		val_err = classification_error(SVM_classifier(X_val, model), y_val);
		train_err(fold) = train_err;
		val_err(fold) = val_err;
    end
	c_err_val(k) = mean(val_err);
	c_sigma_err_train(k) = mean(train_err);
	k = k + 1;
end
[~, index] = min(c_err_val);
c_min = C_val(index);

model = SVM_learner(X, y, c_min, kerneltype, r);
test_err = classification_error(SVM_classifier(X_test, model), y_test);
train_err = classification_error(SVM_classifier(X, model), y);
figure;
decision_boundary_SVM(X_test, y_test, model);
%}

% RBF kernel
%{
kerneltype = 'rbf';

c_sigma_err_train = zeros(length(C_val), length(sigma_val));
c_sigma_err_val = zeros(length(C_val), length(sigma_val));
k = 1;
for c = C_val
	l = 1;
	for sigma = sigma_val
		r = sigma;
		train_err = zeros(num_folds, 1);
		val_err = zeros(num_folds, 1);
        cv = cvpartition(size(X, 1), 'KFold', num_folds);
		for fold = 1:num_folds			
			X_train = X(cv.training(fold), :);
            y_train = y(cv.training(fold), :);
            X_val = X(cv.test(fold), :);
            y_val = y(cv.test(fold), :);
			model = SVM_learner(X_train, y_train, c, kerneltype, r);
			train_err = classification_error(SVM_classifier(X_train, model), y_train);
			val_err = classification_error(SVM_classifier(X_val, model), y_val);
			train_err(fold) = train_err;
			val_err(fold) = val_err;
		end	
		c_sigma_err_val(k, l) = mean(val_err);
		c_sigma_err_train(k, l) = mean(train_err);
		l = l + 1;
	end
	k = k + 1;
end

[value, index] = min(c_sigma_err_val);
[~, idx2] = min(value);
c_min = C_val(index(idx2));
sigma_min = sigma_val(idx2);
model = SVM_learner(X, y, c_min, kerneltype, sigma_min);
test_err = classification_error(SVM_classifier(X_test, model), y_test);
train_err = classification_error(SVM_classifier(X, model), y);
figure;
decision_boundary_SVM(X_test, y_test, model);
%}