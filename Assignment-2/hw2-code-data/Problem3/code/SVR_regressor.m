function [ pred ] = SVR_regressor( X )

[row, col]=size(X);
pred=zeros(row,1);

model=importdata('model.mat');
sv=size(model.alpha,2);

for i=1:row
    z=0;
    for j=1:sv
        x = model.support_vectors(j).x;
        alpha = model.alpha(j)-model.alphastar(j);
        z = z + alpha * compute_kernel(  x , X(i,:) , model.kerneltype , model.r );
    end
    pred(i) = z + model.b;
end

end

