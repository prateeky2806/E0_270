function labels = SVM_classifier(testdata, model)
    % INPUT
    % testdata - m X n matrix of the test data samples
    % model    - SVM model structure returned by SVM_learner
    
    % OUTPUT
    % labels - m x 1 vector of predicted labels
    
    % Write code here
    
	support_vectors = model.support_vectors;
	alphas = model.alphas;
	b = model.b;
	kerneltype = model.kerneltype;
    r = model.r;
	y = model.trainlabels;
	
    K = compute_kernel(support_vectors, testdata, kerneltype, r);
	y_hat = sum(repmat((alphas .* y), [1, size(K, 2)]) .* K, 1)' + repmat(b, [size(testdata, 1), 1]);
    labels = int32((y_hat >= 0));
	labels(find(labels == 0), 1) = -1;
end
