function [output] = EDMatrix( A,B )
[rowA,colA]=size(A);
[rowB,colB]=size(B);
                 
for k=1:colA
    C{k}= repmat(A(:,k),1,rowB);
    D{k}= repmat(B(:,k),1,rowA);
end
S=zeros(rowA,rowB);
for k=1:colA
S=S+(C{k}-D{k}').^2;
end

output=sqrt(S);

end

