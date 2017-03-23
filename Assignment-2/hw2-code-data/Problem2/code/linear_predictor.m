function [ output] = linear_predictor( X , filename )
beta=load(filename);
X(end + 1)=1;
output=X*beta;
end

