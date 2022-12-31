
import numpy as np
import pandas as pd
p = np.mat('8 7 6 8;7 8 8 7') #每一行代表一个对象的指标评分
#A为自己构造的输入判别矩阵
A = np.array([[1,3,1,1/3],[1/3,1,1/2,1/5],[1,2,1,1/3],[3,5,3,1]])
#查看行数和列数
[m,n] = A.shape
 
#求特征值和特征向量
V,D = np.linalg.eig(A)
print('特征值：')
print(V)
print('特征向量：')
print(D)
#最大特征值
tzz = np.max(V)
# print(tzz)
#最大特征向量
k=[i for i in range(len(V)) if V[i] == np.max(V)]
tzx = -D[:,k]
# print(tzx)
 
# #赋权重
quan=np.zeros((n,1))
for i in range(0,n):
    quan[i]=tzx[i]/np.sum(tzx)
Q=quan
# print(Q)
 
#一致性检验
CI=(tzz-n)/(n-1)
RI=[0,0,0.58,0.9,1.12,1.24,1.32,1.41,1.45,1.49,1.52,1.54,1.56,1.58,1.59]
#判断是否通过一致性检验
CR=CI/RI[n-1]
if CR>=0.1:
    print('没有通过一致性检验\n')
else:
    print('通过一致性检验\n')
 
#显示出所有评分对象的评分值
score=p*Q
for i in range(len(score)):
    print('object_score {}：'.format(i),float(score[i]))
