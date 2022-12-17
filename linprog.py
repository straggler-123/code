from scipy import optimize
import pulp
import numpy as np
#简单的题目一维变量
#法一:调用scipy库中optimize模块的linprog函数
#（约束条件无等号）函数默认取最小值
c = np.array([4, -1])  #目标函数
A = np.array([[-1, 1], [-1, -1]])#限制条件1/2 且符号为<=
b = np.array([5, 0])#<=右侧条件
x = (None, 3) #范围
y = (None, None)
res1 = optimize.linprog(c, A, b, bounds=(x, y))

#约束条件带等号
c = np.array([2,3,-5])
A = np.array([[-2,5,-1],[1,3,1]])
b = np.array([-10,12])
Aeq = np.array([[1,1,1]])
beq = np.array([7])
x1 = (0,None)
x2 = (0,None)
x3 = (0,None)
res2 = optimize.linprog(-c,A,b,Aeq,beq,bounds=(x1,x2,x3))#-c最大值


#法二调用pulp库
def LPproblem():
    Myproblem = pulp.LpProblem(sense=pulp.LpMaximize)
    #定义变量
    x1 = pulp.LpVariable('x1', lowBound=0, upBound=None, cat='Continuous')
    x2 = pulp.LpVariable('x2', lowBound=0, upBound=None, cat='Continuous')#带绝对值的话就改参数范围https://blog.csdn.net/HsinglukLiu/article/details/123123292
    x3 = pulp.LpVariable('x3', lowBound=0, upBound=None, cat='Continuous')
    #目标函数及约束条件
    Myproblem += 2 * x1 + 3 * x2 - 5 * x3
    Myproblem += (x1 + x2 + x3 == 7)
    Myproblem += (2 * x1 - 5 * x2 + x3 >= 10)
    Myproblem += (x1 + 3 * x2 + x3 <= 12)
    Myproblem.solve()
    for i in Myproblem.variables():
        print(i.name, '=', i.varValue)  #Xi的值
    Fx = pulp.value(Myproblem.objective) #函数表达式
    return Fx,x1,x2,x3

#二维变量
#产销问题，运费最少的线性规划问题
import gurobipy as gb
production, a = gb.multidict({'A1': 7, 'A2': 4, 'A3': 9})  # 产地
sales, b = gb.multidict({'B1': 3, 'B2': 6, 'B3': 5, 'B4': 6})  # 销地
route, cost = gb.multidict({
    ('A1', 'B1'): 3,
    ('A1', 'B2'): 11,
    ('A1', 'B3'): 3,
    ('A1', 'B4'): 10,
    ('A2', 'B1'): 1,
    ('A2', 'B2'): 9,
    ('A2', 'B3'): 2,
    ('A2', 'B4'): 8,
    ('A3', 'B1'): 7,
    ('A3', 'B2'): 4,
    ('A3', 'B3'): 10,
    ('A3', 'B4'): 5})  # 单位运价

m = gb.Model("lp4")  # 构建模型
x = {}
for i, j in route:
    x[i, j] = m.addVar(vtype=gb.GRB.INTEGER)  # 构建决策变量x为整数

obj = gb.quicksum(x[i, j] * cost[i, j] for i, j in route)  # 目标函数
m.setObjective(obj)  # 默认最小化. 最大化则添加参数 gb.GRB.MAXIMIZE

x = gb.tupledict(x)
m.addConstrs((x.sum("*", j) == b[j] for j in sales), name="con1")  # 添加约束条件1
m.addConstrs((x.sum(i, "*") == a[i]for i in production), name="con2")  # 添加约束条件2
m.write("lp4.lp")
m.optimize()  # 求解
for v in m.getVars(): #输出
    print("%s %g" % (v.varName, v.x))
print("Obj: %g" % m.objVal)

#匈牙利算法，指派问题（0，1分配）
#高阶玩法：考虑多个代价矩阵，考虑相关性https://blog.csdn.net/weixin_42353399/article/details/103450457
from scipy.optimize import linear_sum_assignment

cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
row_ind, col_ind = linear_sum_assignment(cost)
print(row_ind)  # 开销矩阵对应的行索引
print(col_ind)  # 对应行索引的最优指派的列索引
print(cost[row_ind, col_ind])  # 提取每个行索引的最优指派列索引所在的元素，形成数组
print(cost[row_ind, col_ind].sum())  # 数组求和
#分段函数的整数规划??


