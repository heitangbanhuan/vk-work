# train.py
# 这个文件用梯度下降训练线性回归，并做几个小实验看看不同设置的效果。

import json
import numpy as np
import matplotlib.pyplot as plt

# 从自己写的两个文件里把要用的东西引进来
from model import LinearRegression, closed_form
from dataset import read_csv, standardize, back_to_original, get_batch


# 让图里的中文能正常显示（不然会是方框）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def gradient_descent(model, x, y, lr, epochs, size):
    # 用梯度下降训练模型
    # lr 是学习率，epochs 是训练多少轮，size 是每一批用多少个样本
    losses = []

    # 用来打乱数据的随机数工具，固定种子这样每次跑出来的结果一样
    rng = np.random.default_rng(42)

    for epoch in range(epochs):
        for xb, yb in get_batch(x, y, size, rng):
            # 1. 算这一批的预测值
            # 2. 算梯度
            dw, db = model.grad(xb, yb)

            # 3. 更新参数，往梯度的反方向走一小步
            model.w = model.w - lr * dw
            model.b = model.b - lr * db

        # 这一轮结束后，算出在整个训练集上的损失记下来
        losses.append(model.loss(x, y))

    return losses


def metric(y, y_hat):
    # 算三个评价指标：MSE、RMSE、R2
    mse = ((y - y_hat) ** 2).mean()
    rmse = np.sqrt(mse)

    # R2 = 1 - 残差平方和 / 总平方和
    r2 = 1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    return mse, rmse, r2


# ============================ 第一步：读数据 ============================
x_train, y_train = read_csv("../data/train.csv")
x_test, y_test = read_csv("../data/test.csv")

# 把 x 标准化，这一步很重要，不做的话梯度下降很难收敛
x_train, x_test, mean, std = standardize(x_train, x_test)

print("训练集:", len(x_train), "条")
print("测试集:", len(x_test), "条")


# ============================ 第二步：先看看闭式解 ============================
# 闭式解是理论的正确答案，用来和梯度下降的结果对比
w_best, b_best = closed_form(x_train, y_train)
w_raw, b_raw = back_to_original(w_best, b_best, mean, std)
print("闭式解学到的直线: y = %.4f * x + %.4f" % (w_raw, b_raw))


# ============================ 第三步：用梯度下降训练 ============================
# 这里每一批都用全部 100 个样本（size 写 100），也就是批量梯度下降
# 这样的好处是每一次更新用的信息最全，损失会一条直线地往下掉，很好观察
model = LinearRegression()

losses = gradient_descent(model, x_train, y_train, lr=0.1, epochs=200, size=100)

w, b = model.get_wb()
w_raw, b_raw = back_to_original(w, b, mean, std)
print("梯度下降学到的直线: y = %.4f * x + %.4f" % (w_raw, b_raw))
print("最后几轮的损失:", losses[-3:])


# ============================ 第四步：试试不同的学习率 ============================
# 学习率是最重要的一个设置：太小走得太慢，太大就会直接飞出去
print()
print("不同学习率的结果：")

lr_list = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 1.2]
all_losses = {}

for lr in lr_list:
    # 每个学习率都重新建一个模型，从头开始训练
    m = LinearRegression()
    ls = gradient_descent(m, x_train, y_train, lr=lr, epochs=200, size=100)
    all_losses[lr] = ls

    # 顺便把学到的 w 也打出来，看看有没有学到东西
    w_end, b_end = m.get_wb()
    print("  lr = %-6s 最后损失 = %12.4f  w = %.4f" % (lr, ls[-1], w_end))


# ============================ 第五步：试试不同的批量大小 ============================
# 批量大小指的是每次更新参数用多少个样本
# 用全部样本算出来的梯度最准，但一个 epoch 只能更新 1 次
# 只用一个样本算出来的梯度很抖，但一个 epoch 能更新 100 次
print()
print("不同批量大小的结果：")

size_list = [100, 16, 1]

# 用一个字典把每个批量大小最后的结果记下来，后面存 json 还要用
size_result = {}

for size in size_list:
    m = LinearRegression()
    ls = gradient_descent(m, x_train, y_train, lr=0.05, epochs=200, size=size)

    # 记下来
    size_result[str(size)] = ls[-1]

    # 一个 epoch 里更新了多少次 = 样本数 / 批量大小
    times = len(x_train) // size
    print("  每批 %-4d 个样本，每个 epoch 更新 %-4d 次，最后损失 = %.4f"
          % (size, times, ls[-1]))


# ============================ 第六步：在测试集上看效果 ============================
# 训练时候的损失（训练集）
train_mse = model.loss(x_train, y_train)

# 在测试集上预测，再算指标
y_hat = model.predict(x_test)
mse, rmse, r2 = metric(y_test, y_hat)

print()
print("在测试集上的表现：")
print("  MSE  = %.4f" % mse)
print("  RMSE = %.4f" % rmse)
print("  R2   = %.4f" % r2)


# ============================ 第七步：画三张图 ============================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 第一张：损失随着训练轮数下降
# 纵轴用对数，因为损失从两千多掉到二十五，用普通坐标后半段看不出来
axes[0].plot(losses)
axes[0].set_yscale("log")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("loss")
axes[0].set_title("Loss 下降曲线")
axes[0].grid(alpha=0.3)

# 第二张：不同学习率的损失曲线
# lr 抓 1.2 的时候损失会涨到 1e62 那么大，一画上去其他曲线就全被压成一条直线了
# 所以这张图只画到 0.9 为止，1.2 的结果看上面的打印就行
for lr in lr_list:
    if lr <= 0.9:
        axes[1].plot(all_losses[lr], label="lr=" + str(lr))
axes[1].set_yscale("log")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("loss")
axes[1].set_title("不同学习率的对比")
axes[1].legend()
axes[1].grid(alpha=0.3)

# 第三张：测试集的点和我们学到的直线
# 画图的时候希望横轴是原始的 hours，所以要把标准化的数据变回去
# 标准化时做过 x = (x - mean) / std，反过来就是 x = x标准 * std + mean
x_test_raw = x_test * std + mean

axes[2].scatter(x_test_raw, y_test, label="测试集数据")
axes[2].plot(x_test_raw, w_raw * x_test_raw + b_raw, color="red",
             label="y = %.2f x + %.2f" % (w_raw, b_raw))
axes[2].set_xlabel("hours")
axes[2].set_ylabel("score")
axes[2].set_title("测试集拟合效果")
axes[2].legend()
axes[2].grid(alpha=0.3)

# 调整间距之后保存
plt.tight_layout()
plt.savefig("../results/figure.png", dpi=150)
print()
print("图已保存到 results/figure.png")
plt.show()


# ============================ 第八步：把结果存起来 ============================
# 把这次跑出来的关键数字写进一个 json 文件
# 这样写报告的时候数字可以直接从这里拿，不用手抄，也不会抄错

# 把每个学习率最后的损失整理成一个字典
lr_result = {}
for lr in lr_list:
    lr_result[str(lr)] = all_losses[lr][-1]

# 收集所有要保存的内容
result = {
    "n_train": len(x_train),
    "n_test": len(x_test),
    "mean": mean.item(),
    "std": std.item(),
    "closed_form": {"w": w_best.item(), "b": b_best.item()},
    "gradient_descent": {"w": model.w.item(), "b": model.b.item()},
    "train_loss": train_mse.item(),
    "lr_result": lr_result,
    "size_result": size_result,
    "test_mse": mse,
    "test_rmse": rmse,
    "test_r2": r2,
}

# 存成文件，ensure_ascii=False 是为了让中文正常显示
f = open("../results/result.json", "w", encoding="utf-8")
json.dump(result, f, ensure_ascii=False, indent=2)
f.close()
print("结果已保存到 results/result.json")
