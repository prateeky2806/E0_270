%% K means clustering
clc;
clear;

%% reading and initializing the data

dataset=load('../data/synthetic/syndata2.txt');
k=5;                                            
[row,col]=size(dataset);

for i=1:k 
center(i,:)=dataset(i,:);
end

%% clustering
temp=zeros(row,1);
while true

    distance=EDMatrix(dataset,center);
    
    [val,index]=min(distance,[],2);
    
    if(index==temp)
        break;
    else
        temp=index;
    end
    
    for i=1:k
    group=find(index==i);
    if group
        center(i,:)=mean(dataset(find(index==i)),1);
    end
    end
    
end

color=[1 0 0;0 1 0; 0 0 1; 0.5 0.5 0; 0 0.5 0.5];

hold on;
for i=1:row  %for every row
    c= color(index(i),:);
    plot(dataset(i,1),dataset(i,2),'o','markerfacecolor',c,'markeredgecolor',c);
end
hold off;

