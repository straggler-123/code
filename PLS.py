# 偏最小二乘，求解多对多变量间的对应关系
# https://blog.csdn.net/dongke1991/article/details/126843609（流程代码）
import numpy as np
x1=[191,189,193,162,189,182,211,167,176,154,169,166,154,247,193,202,176,157,156,138]
x2=[36,37,38,35,35,36,38,34,31,33,34,33,34,46,36,37,37,32,33,33]
x3=[50,52,58,62,46,56,56,60,74,56,50,52,64,50,46,62,54,52,54,68]
y1=[5,2,12,12,13,4,8,6,15,17,17,13,14,1,6,12,4,11,15,2]
y2=[162,110,101,105,155,101,101,125,200,251,120,210,215,50,70,210,60,230,225,110]
y3=[60,60,101,37,58,42,38,40,40,250,38,115,105,50,31,120,25,80,73,43]
#-----数据读取
data_raw=np.array([x1,x2,x3,y1,y2,y3])
data_raw=data_raw.T #输入原始数据，行数为样本数，列数为特征数
#-----数据标准化
num=np.size(data_raw,0) #样本个数
mu=np.mean(data_raw,axis=0) #按列求均值
sig=(np.std(data_raw,axis=0)) #按列求标准差
data=(data_raw-mu)/sig #标准化，按列减去均值除以标准差
#-----提取自变量和因变量数据
n=3 #自变量个数
m=3 #因变量个数
x0=data_raw[:,0:n] #原始的自变量数据
y0=data_raw[:,n:n+m] #原始的变量数据
e0=data[:,0:n] #标准化后的自变量数据
f0=data[:,n:n+m] #标准化后的因变量数据
#-----相关矩阵初始化
chg=np.eye(n) #w到w*变换矩阵的初始化
w=np.empty((n,0)) #初始化投影轴矩阵
w_star=np.empty((n, 0)) #w*矩阵初始化
t=np.empty((num, 0)) #得分矩阵初始化
ss=np.empty(0) #或者ss=[]，误差平方和初始化
press=[] #预测误差平方和初始化
Q_h2=np.zeros(n) #有效性判断条件值初始化
#-----求解主成分
for i in range(n): #主成分的总个数小于等于自变量个数
    #-----求解自变量的最大投影w和第一主成分t
    matrix=e0.T@f0@f0.T@e0 #构造矩阵E'FF'E
    val,vec=np.linalg.eig(matrix) #计算特征值和特征向量
    index=np.argsort(val)[::-1] #获取特征值从大到小排序前的索引
    val_sort=val[index] #特征值由大到小排序
    vec_sort=vec[:,index] #特征向量按照特征值的顺序排列
    w=np.append(w,vec_sort[:,0][:,np.newaxis],axis=1) #储存最大特征向量
    w_star=np.append(w_star,chg@w[:,i][:,np.newaxis],axis=1) #计算 w*的取值
    t=np.append(t,e0@w[:,i][:,np.newaxis],axis=1) #计算投影
    alpha=e0.T@t[:,i][:,np.newaxis]/(t[:,i]@t[:,i]) #计算自变量和主成分之间的回归系数
    chg=chg@(np.eye(n)-(w[:,i][:,np.newaxis]@alpha.T)) #计算 w 到 w*的变换矩阵
    e1=e0-t[:,i][:,np.newaxis]@alpha.T #计算残差矩阵
    e0=e1 #更新残差矩阵
    #-----求解误差平方和ss
    beta=np.linalg.pinv(t)@f0 #求回归方程的系数，数据标准化，没有常数项
    res=np.array(f0-t@beta) #求残差
    ss=np.append(ss,np.sum(res**2))#残差平方和
    #-----求解残差平方和press
    press_i=[] #初始化误差平方和矩阵
    for j in range(num):
        t_inter=t[:,0:i+1]
        f_inter=f0
        t_inter_del=t_inter[j,:] #把舍去的第 j 个样本点保存起来,自变量
        f_inter_del=f_inter[j,:] #把舍去的第 j 个样本点保存起来，因变量
        t_inter= np.delete(t_inter,j,axis=0) #删除自变量第 j 个观测值
        f_inter= np.delete(f_inter,j,axis=0) #删除因变量第 j 个观测值
        t_inter=np.append(t_inter,np.ones((num-1,1)),axis=1)
        beta1=np.linalg.pinv(t_inter)@f_inter # 求回归分析的系数,这里带有常数项
        res=f_inter_del-t_inter_del[:,np.newaxis].T@beta1[0:len(beta1)-1,:]-beta1[len(beta1)-1,:] #计算残差
        res=np.array(res)
        press_i.append(np.sum(res**2)) #残差平方和，并存储
    press.append(np.sum(press_i)) #预测误差平方和
    #-----交叉有效性检验，判断主成分是否满足条件
    Q_h2[0]=1
    if i>0:
        Q_h2[i]=1-press[i]/ss[i-1]
    if Q_h2[i]<0.0975:
        print('提出的成分个数 r=',i+1)
        break
#-----根据主成分t计算回归方程的系数
beta_Y_t=np.linalg.pinv(t)@f0 #求Y*关于t的回归系数
beta_Y_X=w_star@beta_Y_t#求Y*关于X*的回归系数
mu_x=mu[0:n] #提取自变量的均值
mu_y=mu[n:n+m] #提取因变量的均值
sig_x=sig[0:n] #提取自变量的标准差
sig_y=sig[n:n+m] #提取因变量的标准差
ch0=mu_y-mu_x[:,np.newaxis].T/sig_x[:,np.newaxis].T@beta_Y_X*sig_y[:,np.newaxis].T#算原始数据回归方程的常数项
beta_target=np.empty((n,0)) #回归方程的系数矩阵初始化
for i in range(m):
    a=beta_Y_X[:,i][:,np.newaxis]/sig_x[:,np.newaxis]*sig_y[i]#计算原始数据回归方程的系数
    beta_target=np.append(beta_target,a,axis=1)
target=np.concatenate([ch0,beta_target],axis=0) #回归方程的系数，每一列是一个方程，每一列的第一个数是常数项
print(target)


# 可调用的函数
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn import datasets 
from sklearn.model_selection import GridSearchCV
import numpy as np

#导入数据集
dataset = datasets.load_linnerud() 

#数据集读取为dataframe
col_names = dataset['feature_names'] + dataset['target_names']
data = pd.DataFrame(data= np.c_[dataset['data'], dataset['target']], columns=col_names)

#训练集
x_train=np.array(data.loc[:,dataset['feature_names']])
y_train=np.array(data.loc[:,dataset['target_names']])

#回归模型，参数
pls_model_setup = PLSRegression(scale=True)
param_grid = {'n_components': range(1, 4)}

#GridSearchCV 自动调参
gsearch = GridSearchCV(pls_model_setup, param_grid)

#在训练集上训练模型
pls_model = gsearch.fit(x_train, y_train)

#预测
pred = pls_model.predict(x_train)

#打印 coef
print('Partial Least Squares Regression coefficients:',pls_model.best_estimator_.coef_)
