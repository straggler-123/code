k=4;
a = rand(30,2) * 2;
b = rand(30,2) * 5;
c = rand(30,2) * 10;
X = [a; b; c];  %需要聚类的数据点
xstart = rand(k,2);  %初始聚类中心
[Idx, Center] = kmeans(X, xstart,k);
plot(X(Idx==1,1), X(Idx==1,2), 'kx'); hold on
plot(X(Idx==2,1), X(Idx==2,2), 'gx');
plot(X(Idx==3,1), X(Idx==3,2), 'bx');
plot(X(Idx==4,1), X(Idx==4,2), 'cx');
plot(Center(:,1), Center(:,2), 'r*'); hold off
grid off;
title('K-means cluster result');

disp('xstart = ');
disp(xstart);
disp('Center = ');
disp(Center);