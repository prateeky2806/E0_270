function [label] = knKmeans(X, init, kn)

n = size(X,2);
K = kn;
last = zeros(1,n);
k = init;
label = ceil(k*rand(1,n));
while any(label ~= last)
   [~,~,last(:)] = unique(label);   % remove empty clusters
    E = sparse(last,1:n,1);
    E = E./sum(E,2);
    T = E*K;
    [val, label] = max(T-dot(T,E,2)/2,[],1);
end

