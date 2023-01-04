#聚类算法
#https://zhuanlan.zhihu.com/p/127013012
#k近邻聚类算法https://blog.csdn.net/qq_44725872/article/details/108939815
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def classification(train_feature, train_label, test_feature):
    '''
    对test_feature进行红酒分类
    :param train_feature: 训练集数据，类型为ndarray
    :param train_label: 训练集标签，类型为ndarray
    :param test_feature: 测试集数据，类型为ndarray
    :return: 测试集数据的分类结果
    '''
    #调用模型
    clf = KNeighborsClassifier()
    #使用模型进行训练
    clf.fit(train_feature,train_label)
    #返回预测结果
    return clf.predict(test_feature)

def score(predict_labels,real_labels):
    '''
    对预测的结果进行打分，仅考虑测试集准确率！！！
    '''
    num = 0.
    lenth = len(predict_labels)
    for i in range(lenth):
        if predict_labels[i] == real_labels[i]:
            num = num + 1
    print("预测准确率：",num / lenth)


#加载红酒数据集
wine_dataset = load_wine()

#对数据集进行拆分，X_train、X_test、y_train、y_test分别代表
#训练集特征、测试集特征、训练集标签和测试集标签
X_train, X_test, y_train, y_test = train_test_split(wine_dataset['data'],wine_dataset['target']
                                ,test_size=0.3)

#这是数据没有标准化直接进行训练和预测的结果
print("未进行数据标准化直接训练的模型")
predict1 = classification(X_train,y_train,X_test)
score(predict1,y_test)

print("\n")

#这是数据标准化后的预测结果

#加载标准化模型
scaler = StandardScaler()

#进行数据标准化
train_data = scaler.fit_transform(X_train)
test_data = scaler.fit_transform(X_test)
print("标准化之后训练的模型")
predict2 = classification(train_data,y_train,test_data)
score(predict2,y_test)

#kmeans聚类算法https://zhuanlan.zhihu.com/p/111180811
#算法已封装好，需要数据预处理
import time
import matplotlib.pyplot as plt
import matplotlib
from  sklearn.cluster import KMeans
from sklearn.datasets import load_iris 
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 获取鸢尾花数据集，特征分别是sepal length、sepal width、petal length、petal width
iris = load_iris() 
X = iris.data[:,2:]  # 通过花瓣的两个特征来聚类
k=3  # 假设聚类为3类
# 构建模型
s=time.time()
km = KMeans(n_clusters=k) 
km.fit(X)
print("用sklearn内置的K-Means算法聚类耗时：",time.time()-s)

label_pred = km.labels_   # 获取聚类后的样本所属簇对应值
centroids = km.cluster_centers_  # 获取簇心

#绘制K-Means结果
# 未聚类前的数据分布
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title("未聚类前的数据分布")
plt.subplots_adjust(wspace=0.5)

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=label_pred, s=50, cmap='viridis')
plt.scatter(centroids[:,0],centroids[:,1],c='red',marker='o',s=100)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title("用sklearn内置的K-Means算法聚类结果")
plt.show()

#主成分分析https://blog.csdn.net/qq_52300431/article/details/123482513
#数据归一化
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
data = mms.fit_transform(data)
#降维
from sklearn.decomposition import PCA
from sklearn import preprocessing
pca = PCA()
pca.fit(data)
pca.components_ #模型的各个特征向量 也叫成分矩阵
pca.explained_variance_  # 贡献方差，即特征根
pca.explained_variance_ratio_ #各个成分各自的方差百分比（贡献率）

pca = PCA(5) #确定5个主成分
pca.fit(data)
low_d = pca.transform(data)# low_d降维后的结果
# 求指标在不同主成分线性组合中的系数
k1_spss = pca.components_ / np.sqrt(pca.explained_variance_.reshape(-1, 1))  #成分得分系数矩阵
j = 0
Weights = []
for j in range(len(k1_spss)):
    for i in range(len(pca.explained_variance_)):
        Weights_coefficient = np.sum(100 * (pca.explained_variance_ratio_[i]) * (k1_spss[i][j])) / np.sum(
            pca.explained_variance_ratio_)
    j = j + 1
    Weights.append(np.float(Weights_coefficient))
print('Weights',Weights)
Weights=pd.DataFrame(Weights)
Weights1 = preprocessing.MinMaxScaler().fit(Weights)
Weights2 = Weights1.transform(Weights)
print('Weights2',Weights2)

#因子分析法https://blog.csdn.net/qq_25990967/article/details/122566533
import pandas as pd
import numpy as np
import math as math
import numpy as np
from numpy import *
from scipy.stats import bartlett
from factor_analyzer import *
import numpy.linalg as nlg
from sklearn.cluster import KMeans
from matplotlib import cm
import matplotlib.pyplot as plt
def main():
    df=pd.read_csv("./data/applicant.csv")
    # print(df)
    df2=df.copy()
    print("\n原始数据:\n",df2)
    del df2['ID']
    # print(df2)
    # 皮尔森相关系数
    df2_corr=df2.corr()
    print("\n相关系数:\n",df2_corr)
    #热力图
    cmap = cm.Blues
    # cmap = cm.hot_r
    fig=plt.figure()
    ax=fig.add_subplot(111)
    map = ax.imshow(df2_corr, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title('correlation coefficient--headmap')
    ax.set_yticks(range(len(df2_corr.columns)))
    ax.set_yticklabels(df2_corr.columns)
    ax.set_xticks(range(len(df2_corr)))
    ax.set_xticklabels(df2_corr.columns)
    plt.colorbar(map)
    plt.show()
    # KMO测度
    def kmo(dataset_corr):
        corr_inv = np.linalg.inv(dataset_corr)
        nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
        A = np.ones((nrow_inv_corr, ncol_inv_corr))
        for i in range(0, nrow_inv_corr, 1):
            for j in range(i, ncol_inv_corr, 1):
                A[i, j] = -(corr_inv[i, j]) / (math.sqrt(corr_inv[i, i] * corr_inv[j, j]))
                A[j, i] = A[i, j]
        dataset_corr = np.asarray(dataset_corr)
        kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))
        kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
        kmo_value = kmo_num / kmo_denom
        return kmo_value
    print("\nKMO测度:", kmo(df2_corr))
    # 巴特利特球形检验
    df2_corr1 = df2_corr.values
    print("\n巴特利特球形检验:", bartlett(df2_corr1[0], df2_corr1[1], df2_corr1[2], df2_corr1[3], df2_corr1[4],
                                  df2_corr1[5], df2_corr1[6], df2_corr1[7], df2_corr1[8], df2_corr1[9],
                                  df2_corr1[10], df2_corr1[11], df2_corr1[12], df2_corr1[13], df2_corr1[14]))
    # 求特征值和特征向量
    eig_value, eigvector = nlg.eig(df2_corr)  # 求矩阵R的全部特征值，构成向量
    eig = pd.DataFrame()
    eig['names'] = df2_corr.columns
    eig['eig_value'] = eig_value
    eig.sort_values('eig_value', ascending=False, inplace=True)
    print("\n特征值\n：",eig)
    eig1=pd.DataFrame(eigvector)
    eig1.columns = df2_corr.columns
    eig1.index = df2_corr.columns
    print("\n特征向量\n",eig1)
    # 求公因子个数m,使用前m个特征值的比重大于85%的标准，选出了公共因子是五个
    for m in range(1, 15):
        if eig['eig_value'][:m].sum() / eig['eig_value'].sum() >= 0.85:
            print("\n公因子个数:", m)
            break
    # 因子载荷阵
    A = np.mat(np.zeros((15, 5)))
    i = 0
    j = 0
    while i < 5:
        j = 0
        while j < 15:
            A[j:, i] = sqrt(eig_value[i]) * eigvector[j, i]
            j = j + 1
        i = i + 1
    a = pd.DataFrame(A)
    a.columns = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5']
    a.index = df2_corr.columns
    print("\n因子载荷阵\n", a)
    fa = FactorAnalyzer(n_factors=5)
    fa.loadings_ = a
    # print(fa.loadings_)
    print("\n特殊因子方差:\n", fa.get_communalities())  # 特殊因子方差，因子的方差贡献度 ，反映公共因子对变量的贡献
    var = fa.get_factor_variance()  # 给出贡献率
    print("\n解释的总方差（即贡献率）:\n", var)
    # 因子旋转
    rotator = Rotator()
    b = pd.DataFrame(rotator.fit_transform(fa.loadings_))
    b.columns = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5']
    b.index = df2_corr.columns
    print("\n因子旋转:\n", b)
    # 因子得分
    X1 = np.mat(df2_corr)
    X1 = nlg.inv(X1)
    b = np.mat(b)
    factor_score = np.dot(X1, b)
    factor_score = pd.DataFrame(factor_score)
    factor_score.columns = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5']
    factor_score.index = df2_corr.columns
    print("\n因子得分：\n", factor_score)
    fa_t_score = np.dot(np.mat(df2), np.mat(factor_score))
    print("\n应试者的五个因子得分：\n",pd.DataFrame(fa_t_score))
    # 综合得分
    wei = [[0.50092], [0.137087], [0.097055], [0.079860], [0.049277]]
    fa_t_score = np.dot(fa_t_score, wei) / 0.864198
    fa_t_score = pd.DataFrame(fa_t_score)
    fa_t_score.columns = ['综合得分']
    fa_t_score.insert(0, 'ID', range(1, 49))
    print("\n综合得分：\n", fa_t_score)
    print("\n综合得分：\n", fa_t_score.sort_values(by='综合得分', ascending=False).head(6))
    plt.figure()
    ax1=plt.subplot(111)
    X=fa_t_score['ID']
    Y=fa_t_score['综合得分']
    plt.bar(X,Y,color="#87CEFA")
    # plt.bar(X, Y, color="red")
    plt.title('result00')
    ax1.set_xticks(range(len(fa_t_score)))
    ax1.set_xticklabels(fa_t_score.index)
    plt.show()
    fa_t_score1=pd.DataFrame()
    fa_t_score1=fa_t_score.sort_values(by='综合得分',ascending=False).head()
    ax2 = plt.subplot(111)
    X1 = fa_t_score1['ID']
    Y1 = fa_t_score1['综合得分']
    plt.bar(X1, Y1, color="#87CEFA")
    # plt.bar(X1, Y1, color='red')
    plt.title('result01')
    plt.show()
if __name__ == '__main__':
    main()

#判别分析https://blog.csdn.net/weixin_45678130/article/details/119349864
#https://blog.csdn.net/pengjian444/article/details/71138003
