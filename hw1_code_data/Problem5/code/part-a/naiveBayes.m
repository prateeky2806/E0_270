function [ W,b ] = naiveBayes( train )
%NAIVEBAYES Summary of this function goes here
%   Detailed explanation goes here
X_train = train(:,1:(end-1)); Y_train = train(:,end);
[m,n] = size(X_train);
%X_train = [ones(m,1) X_train];
Y1 = find(Y_train==1);
Y0 = find(Y_train==-1);
X_pos = X_train(Y1,:);
X_neg = X_train(Y0,:);
[p1,p2]=size(X_pos);
[r1,r2]=size(X_neg);

Prob_pos = (sum(X_pos)+1)/(sum(sum(X_pos))+(p2)+1);
Prob_neg = (sum(X_neg)+1)/(sum(sum(X_neg))+(r2)+1);

PY1 = p1/(p1+r1);
PY0 = r1/(p1+r1);

Intercept1 = sum(log(1-Prob_pos)-log(1-Prob_neg));
Intercept2 = log(PY1)-log(PY0);

b = Intercept1 + Intercept2;
W = log(Prob_pos)-log(1-Prob_pos)+log(1-Prob_neg)-log(Prob_neg);

end

