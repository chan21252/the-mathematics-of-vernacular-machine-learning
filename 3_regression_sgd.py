# 随机梯度下降
# author: cuican
# date: 2023/1/31
# -*- coding:utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

# 读取训练集
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]

# 标准化
mu = train_x.mean()
sigma = train_x.std()


def standardize(x):
    return (x - mu) / sigma


train_x_std = standardize(train_x)

# 初始化theta
theta = np.random.rand(3)
print("theta = ", theta)

"""
1 x0 x0**2
1 x1 x1**2
1 x2 x2**2
"""


def to_matrix(x):
    return np.vstack([np.ones(x.size), x, x ** 2]).T


# 训练数据矩阵
X = to_matrix(train_x_std)
print(X)


# 预测函数
def f(x):
    return np.dot(x, theta)


# 误差函数
def E(x, y):
    return 0.5 * np.sum((f(x) - y) ** 2)

def MSE(x, y):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)


# 误差的差值
diff = 1

# 收敛度
ETA = 1e-3

count = 0
errors = [MSE(X, train_y)]
p = np.random.permutation(X.shape[0])

while diff > 1e-2:
    p = np.random.permutation(X.shape[0])
    for x, y in zip(X[p, :], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]
    count += 1
    log = '第{}次, theta = {}, 差值 = {:.4f}'
    print(log.format(count, theta, diff))

axis_x = np.linspace(-3, 3, 100)
plt.plot(train_x_std, train_y, 'o')
plt.plot(axis_x, f(to_matrix(axis_x)))
plt.show()
