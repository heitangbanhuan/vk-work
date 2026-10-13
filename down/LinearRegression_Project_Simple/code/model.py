# model.py
# 这个文件写线性回归的公式：怎么算预测值、怎么算损失、怎么算梯度。

import numpy as np  # 用来做矩阵和向量计算


def forward(x, w, b):
    # 线性回归的公式：y = w * x + b
    # 这里用矩阵乘法写，x 的形状是 (n, 1)，w 的形状是 (1, 1)，乘出来是 (n, 1)
    y_hat = x @ w + b

    return y_hat


def mse_loss(y_hat, y):
    # 均方误差：把每个样本的误差平方，再求平均
    loss = ((y_hat - y) ** 2).mean()

    return loss


def mse_grad(y_hat, y, x):
    # 均方误差对 w 和 b 的导数
    # 推导：L = (1/n) * 求和 (y_hat - y)^2，而 y_hat = w*x + b
    # 用链式法则对 w 求导是 (2/n) * 求和 (y_hat - y) * x
    # 对 b 求导是 (2/n) * 求和 (y_hat - y)
    n = len(x)

    # 先算误差
    diff = y_hat - y

    # x.T 是 x 的转置，(1, n) 乘 (n, 1) 正好把每个误差乘上对应的 x 再求和
    dw = 2 / n * (x.T @ diff)

    # 对 b 的导数不用乘 x，直接把误差加起来
    db = 2 / n * diff.sum()

    # db 现在是一个数，但 b 是 (1, 1) 的数组，所以用 reshape 把它变回 (1, 1)
    db = db.reshape(1, 1)

    return dw, db


def mae_loss(y_hat, y):
    # 平均绝对误差：误差取绝对值再求平均
    loss = np.abs(y_hat - y).mean()

    return loss


def mae_grad(y_hat, y, x):
    # 平均绝对误差的导数
    # 绝对值求导就是符号函数，(正数给 1，负数给 -1)
    n = len(x)

    diff = y_hat - y

    # np.sign 就是符号函数
    sign = np.sign(diff)

    dw = 1 / n * (x.T @ sign)
    db = 1 / n * sign.sum()
    db = db.reshape(1, 1)

    return dw, db


def huber_loss(y_hat, y, delta=1.0):
    # Huber 损失：误差小的时候像均方误差，误差大的时候像绝对值误差
    r = y_hat - y
    abs_r = np.abs(r)

    # np.where 是一个一个元素判断的，满足条件就用第一个值，不满足就用第二个
    one = 0.5 * r ** 2
    two = delta * (abs_r - 0.5 * delta)
    loss = np.where(abs_r <= delta, one, two).mean()

    return loss


def huber_grad(y_hat, y, x, delta=1.0):
    # Huber 损失的导数：误差小的时候导数是 r，误差大的时候导数是 delta 乘符号
    n = len(x)

    r = y_hat - y

    # 先算出每个样本的导数
    d = np.where(np.abs(r) <= delta, r, delta * np.sign(r))

    dw = 1 / n * (x.T @ d)
    db = 1 / n * d.sum()
    db = db.reshape(1, 1)

    return dw, db


# 把损失函数的名字和它对应的两个函数存在一起，用名字就能查到
loss_dict = {
    "mse": (mse_loss, mse_grad),
    "mae": (mae_loss, mae_grad),
    "huber": (huber_loss, huber_grad),
}


class LinearRegression:
    # 线性回归模型，里面存着 w 和 b 两个参数

    def __init__(self):
        # w 是斜率，b 是截距，一开始都设成 0
        # 形状写成 (1, 1) 是为了和 x 做矩阵乘法
        self.w = np.zeros((1, 1))
        self.b = np.zeros((1, 1))

    def predict(self, x):
        # 用现在的参数算预测值
        return forward(x, self.w, self.b)

    def loss(self, x, y, name="mse"):
        # 用现在的参数算损失
        y_hat = self.predict(x)
        return loss_dict[name][0](y_hat, y)

    def grad(self, x, y, name="mse"):
        # 用现在的参数算梯度
        y_hat = self.predict(x)
        return loss_dict[name][1](y_hat, y, x)

    def get_wb(self):
        # 把 w 和 b 拿出来，item() 可以把 (1,1) 的数组变成普通小数
        return self.w.item(), self.b.item()

    def set_wb(self, w, b):
        # 手动设置 w 和 b
        self.w = np.array([[w]])
        self.b = np.array([[b]])

    def __str__(self):
        # 这样 print(模型) 就能看到参数了
        w, b = self.get_wb()
        return "y = %.4f * x + %.4f" % (w, b)


def closed_form(x, y):
    # 闭式解：一个特征的时候，可以直接用公式一步算出最好的 w 和 b，不用迭代
    # 公式：w = 求和 (x - x平均) * (y - y平均) / 求和 (x - x平均)^2
    x_mean = x.mean()
    y_mean = y.mean()

    x2 = x - x_mean
    y2 = y - y_mean

    w = (x2 * y2).sum() / (x2 ** 2).sum()
    b = y_mean - w * x_mean

    return w, b
