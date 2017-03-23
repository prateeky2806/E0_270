clc;clear all;
X=load('kmlabel.mat');
X=X.index;
Y=load('kkmlabel.mat');
Y=(Y.index)';
label=load('../data/synthetic/syndata2_lab.txt');
RAND(X,label)
RAND(label,Y)
