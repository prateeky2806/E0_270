function [] = SVR_learner(trainData, trainLabel, C, kernelType, r,epsilon)


%% Data declaration and initialization
fid = fopen('exp.txt','w');
X=trainData; 
Y=trainLabel;
[row,col]=size(X);

lb=zeros(2*row,1); 
ub=repmat(C,2*row,1);

Y_=[Y;-Y];
E=repmat(-epsilon,2*row,1);
f = E + Y_ ;

KernelMat=compute_kernel(X,X,kernelType,r);
temp=[KernelMat,-KernelMat;-KernelMat,KernelMat];
temp=-temp;


Aeq=[ones(1,row),-ones(1,row)];
beq=0;

A=zeros(1,2*row);
b=0;

Alfa=quadprog(-temp,-f,A,b,Aeq,beq,lb,ub) ;

Alf=Alfa(1:row);
Alfatar=Alfa(row+1:2*row);

k=1;
for i=1:row
    
       if Alf(i)>0 || Alfatar(i)>0
        support(k).x=X(i,:);
        support(k).y=Y(i);
        support_Alf(k)=Alf(i);
        support_Alfatar(k)=Alfatar(i);
        k=k+1;
    end
end
fprintf(fid,'the number of support vectors are %d',k-1);

% n=size(support,1);
b=0;
k=0;
for i=1:row
    wtx=0;
    if ( Alf(i)>0 && Alf(i)<C )   %i.e. they belong to SV1
        for j=1:row
            if Alf(j)>0 || Alfatar(j)>0  % i.e. this j is support vector
                wtx = wtx + (Alf(j)-Alfatar(j))*K(j,i) ;
            end
        end
        b = b + ( Y(i)-wtx-epsilon );
        k=k+1;
    elseif ( Alfatar(i)>0 && Alfatar(i)<C )
        for j=1:m
            if Alf(j)>0 || Alfatar(j)>0  % i.e. this j is support vector
                wtx = wtx + (Alf(j)-Alfatar(j))*K(j,i) ;
            end
        end
        b = b + ( wtx-Y(i)-epsilon );
        k=k+1;
    end
    
end

b=b/k;

model.b = b;
model.Alf = support_Alf;
model.Alfatar = support_Alfatar;
model.kerneltype = kernelType;
model.r = r;
model.C = C;
model.traindata = X;
model.trainlabel = Y;
model.support_vectors = support;

filename='model';
save(filename,'model');

end
