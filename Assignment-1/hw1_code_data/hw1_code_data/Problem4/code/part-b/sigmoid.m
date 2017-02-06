function Z = sigmoid(z)
Z=zeros(size(z));
Z=1./(1+exp(-z));
end
