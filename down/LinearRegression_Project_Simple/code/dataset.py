# dataset.py
# 这个文件用来读数据，并且把数据整理成模型能用的样子。

import csv          # 用来读 csv 文件
import numpy as np  # 用来做矩阵计算

# 两个数据文件的路径
train_path = "../data/train.csv"
test_path = "../data/test.csv"


def read_csv(path):
    # 打开文件
    f = open(path, "r", encoding="utf-8")

    # csv.reader 会把每一行切成一个列表，比如 ["2.21", "53.14"]
    reader = csv.reader(f)

    # 第一行是表头，先读掉它
    header = next(reader)

    # 准备两个列表装数据
    x = []
    y = []

    # 一行一行地读
    for row in reader:
        # row[0] 是学习时长，row[1] 是成绩，都是字符串，要用 float 转成小数
        x.append(float(row[0]))
        y.append(float(row[1]))

    # 关掉文件
    f.close()

    # 把列表变成 numpy 数组，方便后面做矩阵运算
    x = np.array(x)
    y = np.array(y)

    # 数组的形状要是 (样本数, 1)，所以用 reshape 把每个数单独放一行
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return x, y


def standardize(x_train, x_test):
    # 求训练集的平均值和标准差
    mean = x_train.mean()
    std = x_train.std()

    # 把训练集和测试集都变成“平均值是 0，标准差是 1”的数据
    # 注意：测试集要用训练集的 mean 和 std，这样才公平
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return x_train, x_test, mean, std


def back_to_original(w, b, mean, std):
    # 模型是在标准化后的数据上训练的，所以 w 和 b 也是在那个尺度下的
    # 这个函数把它们换回用原始 hours 就能直接算的形式
    # 推导：y = w * (x - mean) / std + b，整理一下就是下面两行
    w_new = w / std
    b_new = b - w * mean / std

    return w_new, b_new


def get_batch(x, y, size, rng):
    # 把样本的编号打乱，这样每个 epoch 拿到的批次都不一样
    index = rng.permutation(len(x))

    # 按照 size 一批一批地取出来
    for i in range(0, len(x), size):
        # 取出这一批的编号
        this_index = index[i:i + size]

        # 用编号把对应的数据挑出来
        xb = x[this_index]
        yb = y[this_index]

        # yield 表示“交出这一批，然后停在这里，下次接着往下走”
        yield xb, yb


def prepare_data():
    # 读训练集和测试集
    x_train, y_train = read_csv(train_path)
    x_test, y_test = read_csv(test_path)

    # 把特征标准化
    x_train, x_test, mean, std = standardize(x_train, x_test)

    # 返回训练和测试要用的数据
    return x_train, y_train, x_test, y_test, mean, std


# 直接运行这个文件的时候，打印一些信息看看读对了没有
x_train, y_train, x_test, y_test, mean, std = prepare_data()
print("训练集样本数:", len(x_train))
print("测试集样本数:", len(x_test))
print("训练集的平均值:", mean)
print("训练集的标准差:", std)
print("训练集前 5 条数据:")
for i in range(5):
    print("   ", x_train[i], y_train[i])
