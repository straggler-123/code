clear,clc
rng(0)

x1 = normrnd(5,1,100,1);
x2 = normrnd(6,1,100,1);
figure
boxplot([x1,x2],'Notch','on','Labels',{'mu = 5','mu = 6'}) 
title('Compare Random Data from Different Distributions')

