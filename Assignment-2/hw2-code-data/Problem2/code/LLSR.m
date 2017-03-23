function [ output ] = LLSR( X , Y)
[row , col]=size(X);
X=[X ones(row,1)];
fid=fopen('weights_LLS.txt','w');
output=inv((X'*X))*(X'*Y);
fprintf(fid, '%f\n', output);
fclose(fid);
end

