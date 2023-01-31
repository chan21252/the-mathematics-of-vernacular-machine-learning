# 线性回归
# author: chan
# date: 2023/01/28

import numpy as np
import matplotlib.pyplot as plt

# 读取训练集
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]

plt.plot(train_x, train_y, 'o')
plt.show()

theta0 = np.random.rand()
theta1 = np.random.rand()


# 线性回归函数
def f(x):
    return theta0 + theta1 * x


# 误差函数
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)


# 标准化
mu = train_x.mean()
sigma = train_x.std()


def standardize(x):
    return (x - mu) / sigma


train_x_std = standardize(train_x)

# theta收敛度
ETA = 1e-3
# 初始误差
error = E(train_x_std, train_y)
# 前后两次Error误差的差值
diff = 1
# 更新次数
count = 0

while diff > 1e-2:
    tmp0 = theta0 - ETA * np.sum(f(train_x_std) - train_y)
    tmp1 = theta1 - ETA * np.sum((f(train_x_std) - train_y) * train_x_std)
    theta0 = tmp0
    theta1 = tmp1
    error_current = E(train_x_std, train_y)
    diff = error - error_current
    error = error_current
    count = count + 1
    log = '第{}次, theta0 = {:.3f}, theta1 = {:.3f}, 差值 = {:.4f}'
    print(log.format(count, theta0, theta1, diff))

axis_x = np.linspace(-3, 3, 100)
plt.plot(train_x_std, train_y, 'o')
plt.plot(axis_x, f(axis_x))
plt.show()

print(f(standardize(100)))
print(f(standardize(200)))
print(f(standardize(300)))
