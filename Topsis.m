clc,clear
a=[0.1 5 5000 4.7;
0.2 6 6000 5.6;
0.4 7 7000 6.7;
0.9 10 10000 2.3;
1.2 2 400 1.8];
qujian=[5,6];lb=2;ub=12;
xx=2;
w=[0.2 0.3 0.4 0.1];
%以上为超参。区间为最优区间，lb ub为不能容忍的上下限,xx为区间型属性的序号,w为权重向量。

[m,n]=size(a);
fun=@(qujian,lb,ub,x)(1-(qujian(1)-x)./(qujian(1)-lb)).*(x>=lb&x<qujian(1))+...
(x>=qujian(1)&x<=qujian(2))+(1-(x-qujian(2))./(ub-qujian(2))).*...
(x>qujian(2)&x<=ub);

a(:,xx)=fun(qujian,lb,ub,a(:,xx));%对属性xx进行变换
for j=1:n
b(:,j)=a(:,j)/norm(a(:,j));%向量规范化
end

c=b.*repmat(w,m,1);%求加权矩阵
%repmat(A,m,n)%将A复制m*n块
cstar=max(c);%求正理想解

%注意：以下是对成本型的正理想解的修正

cstar(4) = min(c(:,4));%属性4为成本型

c0=min(c);%求负理想解

%注意：以下是对成本型的负理想解的修正

c0(4)=max(c(:,4));%属性4为成本型的
for i=1:m
sstar(i)=norm(c(i,:)-cstar);%求到正理想解的距离
s0(i)=norm(c(i,:)-c0);%求到负理想解的距离
end
f=s0./(sstar+s0);
[sf,ind]=sort(f,'descend') %求排序结果
% "ascend"时,进行升序排序,为"descend "时,进行降序排序
