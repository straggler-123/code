#动态规划问题：建模的时候需要整出二维表的行/列索引（阶段与状态）https://blog.csdn.net/XianZhe_/article/details/114962984
# 递归算法


import numpy as np
#背包问题
# 行李数n，不超过的重量W，重量列表w和价值列表p
def fun(n, W, w, p):
    a = np.array([[0] * (W + 1)] * (n + 1))
    # 依次计算前i个行李的最大价值，n+1在n的基础上进行
    for i in range(1, n + 1):
        for j in range(1, W + 1):
            if w[i - 1] > j:
                a[i, j] = a[i - 1, j]
            else:
                a[i, j] = max(a[i - 1, j], p[i - 1] + a[i - 1, j - w[i - 1]])  # 2种情况取最大值
    # print(a)
    print('max value is' + str(a[n, W]))
    findDetail(p, n, a[n, W])


# 找到价值列表中的一个子集，使得其和等于前面求出的最大价值，即为选择方案
def findDetail(p, n, v):
    a = np.array([[True] * (v + 1)] * (n + 1))
    for i in range(0, n + 1):
        a[i][0] = True
    for i in range(1, v + 1):
        a[0][i] = False
    for i in range(1, n + 1):
        for j in range(1, v + 1):
            if p[i - 1] > j:
                a[i, j] = a[i - 1, j]
            else:
                a[i, j] = a[i - 1, j] or a[i - 1, j - p[i - 1]]
    if a[n, v]:
        i = n
        result = []
        while i >= 0:
            if a[i, v] and not a[i - 1, v]:
                result.append(p[i - 1])
                v -= p[i - 1]
            if v == 0:
                break
            i -= 1
        print(result)
    else:
        print('error')


weights = [1, 2, 5, 6, 7, 9]
price = [1, 6, 18, 22, 28, 36]
fun(len(weights), 13, weights, price)


def dynamic_p() -> list:
    items = [  									 # 物品项
        {"name": "水", "weight": 3, "value": 10},
        {"name": "书", "weight": 1, "value": 3},
        {"name": "食物", "weight": 2, "value": 9},
        {"name": "小刀", "weight": 3, "value": 4},
        {"name": "衣物", "weight": 2, "value": 5},
        {"name": "手机", "weight": 1, "value": 10}
    ]
    max_capacity = 6                             # 约束条件为 背包最大承重为6
    dp = [[0] * (max_capacity + 1) for _ in range(len(items) + 1)]

    for row in range(1, len(items) + 1):         # row 代表行
        for col in range(1, max_capacity + 1):   # col 代表列
            weight = items[row - 1]["weight"]    # 获取当前物品重量
            value = items[row - 1]["value"]      # 获取当前物品价值
            if weight > col:                     # 判断物品重量是否大于当前背包容量
                dp[row][col] = dp[row - 1][col]  # 大于直接取上一次最优结果 此时row-1代表上一行
            else:
                # 使用内置函数max()，将上一次最优结果 与 当前物品价值+剩余空间可利用价值 做对比取最大值
                dp[row][col] = max(value + dp[row - 1][col - weight], dp[row - 1][col])
    return dp


dp = dynamic_p()
for i in dp:                                     # 打印数组
    print(i)

print(dp[-1][-1])                                # 打印最优解的价值和


# 上台阶问题，（共n级，一次只能1/2级，求多少步）
class Solution:
    """
    @param n: an integer
    @return: an ineger f(n)
    """

    def up(self, n):
        # f(n)=f(n-1)+f(n-2)
        L = []
        L.append(1)
        L.append(2)
        for i in range(2, n):
            L.append(L[i - 1] + L[i - 2])
        return L[n - 1]


# 俄罗斯信封套娃（需要排序，上面的问题答案与顺序无关）
def max_envelopes(envelopes: list) -> int:
    if not envelopes:
        return 0
    length = len(envelopes)
    # 排列规则：对宽度 w 进行升序排序，如果 w 相同时，则按高度 h 降序排序
    # 全升序排列
    envelopes.sort(key=lambda x: (x[0], x[1]))
    dp = [1] * length
    # 小优化，从1开始循环，dp[0]为最小情况1,即只有自己一个的最长递增子序列
    for i in range(1, length):
        for k in range(i):
            if envelopes[i][1] > envelopes[k][1] and envelopes[i][0] > envelopes[k][0]:
                dp[i] = max(dp[i], dp[k] + 1)
    return max(dp)
