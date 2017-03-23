function [output]=LRR( X, Y, lambda )

[row,col]=size(X);
X=[X ones(row,1)];

beta=inv((X'*X+eye(col+1)*lambda))*(X'*Y);
output=beta;
fid=fopen('weights_RR.txt','w');
fprintf(fid, '%f\n', beta);
fclose(fid);
end

