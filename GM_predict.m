%灰色预测步骤
%（1）输入前期的小样本数据
%（2）输入预测个数
%（3）运行
clear,clc
%y=input('请输入数据');
[y]=xlsread('C:\Users\Lenovo\Desktop\mm.xls',-1);
n=length(y);
num=0;
for mm=2:n
    if exp(-2/(n+1))<(y(mm-1)/y(mm))&&(y(mm-1)/y(mm))<exp(2/(n+1))
    else
        num=num+1;
    end   
end
disp('没有通过级比检验个数为');disp(num);
yy=ones(n,1);
yy(1)=y(1);
for i=2:n
    yy(i)=yy(i-1)+y(i);
end
B=ones(n-1,2);
for i=1:(n-1)
    B(i,1)=-(yy(i)+yy(i+1))/2;
    B(i,2)=1;
end
BT=B';
for j=1:(n-1)
    YN(j)=y(j+1);
end
YN=YN';
A=inv(BT*B)*BT*YN;
a=A(1);
u=A(2);
t=u/a;
t_test=input('输入需要预测的个数');
i=1:t_test+n;
yys(i+1)=(y(1)-t).*exp(-a.*i)+t;
yys(1)=y(1);
for j=n+t_test:-1:2
    ys(j)=yys(j)-yys(j-1);
end
x=1:n;
xs=2:n+t_test;
yn=ys(2:n+t_test);
plot(x,y,'^r',xs,yn,'*-b');
flag1=0;flag2=0;
for i=1:n-1
    det=(y(i)-yn(i))/y(i);
    if abs(det)<0.1
        flag1=flag1+1;
    elseif abs(det)<0.2
        flag2=flag2+1;
    end
end
disp(['残差检验后，有',num2str(100*flag1/(n-1)),'%通过' ...
    '高精度检验，有',num2str(100*(flag2+flag1)/(n-1)),'%通过低精度检验']);
    disp(['预测值为：',num2str(ys(n+1:n+t_test))]);
