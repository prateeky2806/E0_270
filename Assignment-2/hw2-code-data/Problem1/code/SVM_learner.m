function model = SVM_learner(traindata, trainlabels, C, kerneltype, r)
    % INPUT : 
    % traindata   - m X n matrix, where m is the number of training points
    % trainlabels - m X 1 vector of training labels for the training data
    % C           - SVM regularization parameter (positive real number)
    % kerneltype  - one of strings 'linear', 'poly', 'rbf'
    %               corresponding to linear, polynomial, and RBF kernels
    %               respectively.
    % r           - integer parameter indicating the degree of the
    %               polynomial for polynomial kernel, or the width
    %               parameter for the RBF kernel; not used in the case of
    %               linear kerne and can be set to a default value.
    
    % OUTPUT
    % returns the structure 'model' which has the following fields, in
    % addition to the training data/parameters.(You can choose to add more
    % fields to this structure needed for your implementation)
    
    
    % 	alphas      	- m X 1 vector of support vector coefficients
    % 	b           	- SVM bias term
    % 	objective   	- optimal objective value of the SVM solver
    % 	support_vectors - the subset of training data, which are the support vectors
    
    % Default code below. Fill in the code for solving the
    % SVM dual optimization problem using quadprog function
    
    b = 0;
    objective = 0;
    alphas = repmat(0, size(traindata, 1), 1);
    
    X = traindata;
    y = trainlabels;
    K = compute_kernel(X, X, kerneltype, r);	% kernel matrix
	y_pairs = y * y';
	eqn_one = (K .* y_pairs);
	q = -ones(size(alphas));
    eqn_two = y';
	eqn_three = 0;
    
	% upper and lower bound for x
	upper_bound = double(C*ones(size(alphas)));
	lower_bound = zeros(size(alphas));
	options = optimoptions('quadprog', 'Display', 'off');
	[alphas, objective] = quadprog(eqn_one, q, [], [], eqn_two, eqn_three, lower_bound, upper_bound, [],...
															options);
	% bais term b
    y_hat = sum(repmat((alphas .* y), [1, size(K, 2)]) .* K, 1)';
	index = find(alphas > 0 & alphas < C);
    b = mean((1.0 ./ y(index, :)) - y_hat(index, :));
    
    model.b = b;
    model.objective = objective;
    model.alphas = alphas; 
    model.kerneltype = kerneltype;
    model.r = r;
    model.C = C;
    model.traindata = traindata;
    model.trainlabels = trainlabels;
    model.support_vectors = traindata;
end
