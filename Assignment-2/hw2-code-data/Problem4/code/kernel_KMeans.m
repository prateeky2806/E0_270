clc;clear ALL;
dataset=load('../data/synthetic/syndata2.txt');
KernelMat=load('../data/synthetic/syndata2_kernel.txt');
k=5;                                            
[row,col]=size(dataset);
[index] = knKmeans(dataset',k,KernelMat);
color=[1 0 0;0 1 0; 0 0 1; 0.5 0.5 0; 0 0.5 0.5];
hold on;
for i=1:row
    c= color(index(i),:);
    plot(dataset(i,1),dataset(i,2),'o','markerfacecolor',c,'markeredgecolor',c);
end
hold off;
