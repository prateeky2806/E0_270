clear all; close all; clc;


train_file = '../data/Spambase/train.txt';
test_file = '../data/Spambase/test.txt';
dir = '../data/Spambase-cross-validation/';
C_val = [1, 10, 100, 1000, 10000];
num_folds = 5;
kerneltype = 'linear';
r = 0;

% Read Data
imp_features = 1:57;
X = load(train_file);
y = X(:, 58);
X = X(:, imp_features);
X_test = load(test_file);
y_test = X_test(:, 58);
X_test = X_test(:, imp_features);
m_test = size(X_test, 1);
n_test = size(X_test, 2);
m = size(X, 1);
n = size(X, 2);
cross_err_train = zeros(length(C_val), 1);
cross_err_val = zeros(length(C_val), 1);
k = 1;
for c = C_val
	train_err = zeros(num_folds, 1);
	val_err = zeros(num_folds, 1);
	for fold = 1:num_folds		
		X_train = load([dir 'Fold' num2str(fold) '/cv-train.txt']);
		y_train = X_train(:, 58);
		X_train = X_train(:, imp_features);
		X_val = load([dir 'Fold' num2str(fold) '/cv-test.txt']);
		y_val = X_val(:, 58);
		X_val = X_val(:, imp_features);
		
		svm_model = SVM_learner(X_train, y_train, c, kerneltype, r);
		% Calculate errors
		train_err = classification_error(SVM_classifier(X_train, svm_model), y_train);
		val_err = classification_error(SVM_classifier(X_val, svm_model), y_val);
		
		% Record the data
		train_err(fold) = train_err;
		val_err(fold) = val_err;
    end	
	cross_err_val(k) = mean(val_err);
	cross_err_train(k) = mean(train_err);
	k = k + 1;
end

[~, index1] = min(cross_err_val);
c_min = C_val(index1);

svm_model = SVM_learner(X, y, c_min, kerneltype, r);
test_err = classification_error(SVM_classifier(X_test, svm_model), y_test);

figure;
hold on;
plot(log10(C_val), cross_err_val, 'g-', 'linewidth', 1.0);
plot(log10(C_val), cross_err_train, 'c-', 'linewidth', 1.0);
hold off;