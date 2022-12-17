#基础的非线性规划代码参考资料：https://zhuanlan.zhihu.com/p/393509520
#导包：scipy的optimize模块（这个包真的牛逼）
from scipy.optimize import brent, minimize, fmin
import numpy as np
#单变量无约束优化问题
def objf(x):  # 目标函数
    fx = x**2 - 8*np.sin(2*x+np.pi)
    return fx

xIni = -5.0
xOpt= brent(objf, brack=(xIni,2))
print("xIni={:.4f}\tfxIni={:.4f}".format(xIni,objf(xIni)))
print("xOpt={:.4f}\tfxOpt={:.4f}".format(xOpt,objf(xOpt)))

# 多变量无约束优化问题(Scipy.optimize.brent)
def objf2(x):  # Rosenbrock benchmark function
    fx = sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    return fx
xIni = np.array([-2, -2])
xOpt = fmin(objf2, xIni)
print("xIni={:.4f},{:.4f}\tfxIni={:.4f}".format(xIni[0],xIni[1],objf2(xIni)))
print("xOpt={:.4f},{:.4f}\tfxOpt={:.4f}".format(xOpt[0],xOpt[1],objf2(xOpt)))


#多变量边界约束优化问题(Scipy.optimize.minimize)
def objf3(x):  # Rosenbrock 测试函数
    fx = sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    return fx
# 定义边界约束（优化变量的上下限）
b0 = (0.0, None)  # 0.0 <= x[0] <= Inf
b1 = (0.0, 10.0)  # 0.0 <= x[1] <= 10.0
b2 = (-5.0, 100.)  # -5.0 <= x[2] <= 100.0
bnds = (b0, b1, b2)  # 边界约束
# 优化计算
xIni = np.array([1., 2., 3.])
resRosen = minimize(objf3, xIni, method='SLSQP', bounds=bnds)
xOpt = resRosen.x
print("xOpt = {:.4f}, {:.4f}, {:.4f}".format(xOpt[0],xOpt[1],xOpt[2]))
print("min f(x) = {:.4f}".format(objf3(xOpt)))

# 约束非线性规划问题(Scipy.optimize.minimize)
def objF4(x):  # 定义目标函数
    a, b, c, d = 1, 2, 3, 8
    fx = a*x[0]**2 + b*x[1]**2 + c*x[2]**2 + d
    return fx

# 定义约束条件函数
def constraint1(x):  # 不等式约束 f(x)>=0
    return x[0]** 2 - x[1] + x[2]**2
def constraint2(x):  # 不等式约束 转换为标准形式
    return -(x[0] + x[1]**2 + x[2]**3 - 20)
def constraint3(x):  # 等式约束
    return -x[0] - x[1]**2 + 2
def constraint4(x):  # 等式约束
    return x[1] + 2*x[2]**2 -3

# 定义边界约束
b = (0.0, None)
bnds = (b, b, b)

# 定义约束条件
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'eq', 'fun': constraint3}
con4 = {'type': 'eq', 'fun': constraint4}
cons = ([con1, con2, con3,con4])  # 约束条件中不等式的要求：>=

# 求解优化问题
x0 = np.array([1., 2., 3.])  # 定义搜索的初值
res = minimize(objF4, x0, method='SLSQP', bounds=bnds, constraints=cons)   #梯度的求解方法（method）按照题目来确定

print("Optimization problem (res):\t{}".format(res.message))  # 优化是否成功
print("xOpt = {}".format(res.x))  # 自变量的优化值
print("min f(x) = {:.4f}".format(res.fun))  # 目标函数的优化值

