# -*- coding: utf-8 -*-
# =============================================================================
# train.py  ——  训练脚本：用梯度下降训练线性回归，并做一系列对比实验
# -----------------------------------------------------------------------------
# 这个文件是整个项目的"总指挥"，它把 dataset.py（数据）和 model.py（模型）串起来：
#
#     1. 读数据（train.csv 用来训练，test.csv 用来测试）
#     2. 实现梯度下降优化算法（BGD / SGD / MBGD，外加动量和学习率衰减）
#     3. 跑实验：
#          实验 A：观察 loss 随迭代次数的下降曲线 + 拟合直线如何一步步变好
#          实验 B：不同学习率的收敛情况对比（以及"学习率太大会发散"）
#          实验 C：特征标准化 vs 不标准化（为什么一定要做标准化）
#          实验 D：批量大小对比（BGD / Mini-batch / SGD）
#          实验 E：不同损失函数（MSE / MAE / Huber）对异常值的鲁棒性
#          实验 F：动量 + 学习率衰减的效果
#          实验 G：最终模型在测试集上的拟合效果
#     4. 把所有曲线和图保存到 results/ 文件夹（report.pdf 里直接用这些图）
#
# 运行方式（在项目根目录下）：
#     python code/train.py                     # 默认跑到 MLP 那个位置？不，跑全部实验
#     python code/train.py --epochs 300        # 自定义收敛实验的迭代轮数
#     python code/train.py --no-figures        # 只算数、不画图（服务器上快速验证用）
# =============================================================================

# ---------------------------- 0. 先把环境准备好 ----------------------------
# os 用来拼路径、建文件夹；sys 用来把 code/ 目录加进模块搜索路径。
import os
import sys

# argparse 用来解析命令行参数，这样同一个脚本可以有不同的运行方式。
import argparse

# csv 用来把实验结果写成表格文件（Python 自带，不需要额外安装）。
import csv

# json 用来把最终的指标数字存成文件，方便回头查、也方便 report 生成脚本读取。
import json

# 下面这几行必须在 import matplotlib 之前执行：
# 1) 强制使用 "Agg" 后端 —— 它表示"只把图存成文件，不弹窗显示"，
#    在没有图形界面的服务器上也能正常出图。
os.environ.setdefault("MPLBACKEND", "Agg")

# 2) 把 matplotlib 的缓存目录指到项目内，避免系统用户目录没有写权限时报错
#    （这是一个真实踩过的坑，详见 note.md）。
_HERE = os.path.dirname(os.path.abspath(__file__))              # 当前文件所在目录 code/
_PROJECT_ROOT = os.path.dirname(_HERE)                          # 项目根目录
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_PROJECT_ROOT, ".mplcache"))

# 3) 把 code/ 目录加入模块搜索路径，这样无论从哪里运行脚本，
#    都能 import 到同目录下的 dataset.py 和 model.py。
sys.path.insert(0, _HERE)

# numpy：数值计算。
import numpy as np

# matplotlib.pyplot：画图。as plt 是约定俗成的简写。
import matplotlib.pyplot as plt

# 从自己写的两个模块里导入需要的函数和类。
# 这里刻意"用到什么就导入什么"，而不是 import *，这样一眼能看出依赖了什么。
from dataset import prepare_data, iterate_minibatches, restore_params, to_column
from model import LOSS_REGISTRY, LinearRegression, fit_closed_form, mse_loss

# ---------------------------- 1. 全局设置 ----------------------------

# 结果保存目录：项目根目录/results
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")

# 图片分辨率。dpi 越高越清晰，但文件也越大。150 打印在报告里已经足够清楚。
FIGURE_DPI = 150

# 画图时用统一的一套颜色，保证报告里 8 张图的风格一致。
C_TRAIN = "#1f77b4"      # 蓝色：训练集
C_TEST = "#d62728"       # 红色：测试集
C_LINE = "#2ca02c"       # 绿色：模型拟合的直线
C_GRAY = "#7f7f7f"       # 灰色：辅助线

# 是否把图片写进 results/ 目录。
# 默认是 True；如果在命令行加了 --no-figures，main() 会把它改成 False，
# 这样脚本就只跑训练和统计、不落盘图片（在服务器上快速验证时能省不少时间）。
SAVE_FIGURES = True


def setup_matplotlib_for_chinese():
    """
    让 matplotlib 能正确显示中文，并且把负号显示成正常的减号。

    背景知识：matplotlib 默认字体不含中文字形，直接画中文会显示成一堆方框（□□□）。
    解决办法是把字体设成系统里带中文的字体。Windows 上常用 SimHei（黑体）、
    Microsoft YaHei（微软雅黑）。
    """

    # 按优先级列出候选字体名。matplotlib 会挑第一个能找到的。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]

    # 如果不设这一行，坐标轴上的负号会显示成方框（因为默认字体里没有 Unicode 减号）。
    plt.rcParams["axes.unicode_minus"] = False

    # 图片里的字号稍微调大一点，方便打印阅读。
    plt.rcParams["font.size"] = 11


def set_seed(seed):
    """
    固定随机种子，保证实验"可复现"。

    为什么重要？因为 SGD / Mini-batch 会随机打乱数据顺序，
    如果不固定种子，每次运行的结果都会略有不同，实验就没法对比了。
    这在科研和工程里是一条铁律：任何带随机性的实验，都必须固定种子。
    """

    # numpy 的全局随机种子（本项目主要用 default_rng，这里设置是为了双保险）。
    np.random.seed(seed)

    # 返回一个独立的随机数生成器，供本脚本显式使用，做到"谁用谁负责"。
    return np.random.default_rng(seed)


# ---------------------------- 2. 把输出同时写到日志文件 ----------------------------

class Tee(object):
    """
    一个"一分为二"的小工具：让 print 的内容既显示在屏幕上，也写进日志文件。

    为什么需要它？
        训练过程要跑一两个小时的时候，我们希望事后还能回头查"当时到底输出了什么"，
        但终端里的内容一关窗口就没了。有了 Tee，就可以把完整日志存成 txt。

    名字来历：Tee 是 Linux 里的一个命令（也是水管三通接头的名字），
    作用就是把一路输入分成两路输出。
    """

    def __init__(self, *streams):
        """
        参数:
            *streams : 任意多个"像文件一样的对象"（必须有 write 方法）。
                       * 号表示"可变参数"，调用时想传几个就传几个。
        """

        # 把这些输出目标存起来，写的时候挨个写一遍。
        self.streams = streams

    def write(self, text):
        """把 text 写到每一个输出目标里。"""
        # 逐个遍历输出目标。
        for s in self.streams:
            # 写进去。
            s.write(text)
            # flush 表示"立刻写盘/立刻刷新显示"，而不是先攒在缓冲区里。
            # 训练脚本这样处理可以保证万一中途崩溃，日志也是完整的。
            s.flush()

    def flush(self):
        """有些库会调用 flush，我们也必须提供这个方法，否则会报错。"""
        # 逐个刷新。
        for s in self.streams:
            s.flush()


def start_logging(log_path):
    """
    开始把标准输出（print 的内容）同时写进 log_path 这个文件。

    返回:
        一个"原来的 stdout"，程序结束时用 restore_logging 还回去。
    """

    # 以"写"模式打开日志文件，encoding 用 utf-8 保证中文正常。
    log_file = open(log_path, "w", encoding="utf-8")

    # 记下当前的 stdout（通常是终端），以便最后还原。
    original_stdout = sys.stdout

    # 用 Tee 把"终端 + 日志文件"打包成一个新的 stdout。
    sys.stdout = Tee(original_stdout, log_file)

    # 返回这两样东西，方便结束时收拾现场。
    return original_stdout, log_file


def stop_logging(original_stdout, log_file):
    """
    结束日志记录：还原 stdout，关闭日志文件。
    """

    # 先把 stdout 还回去，否则后面的 print 还会想往已经关闭的文件里写，会报错。
    sys.stdout = original_stdout

    # 关掉日志文件。
    log_file.close()


# ---------------------------- 3. 评估指标 ----------------------------

def r2_score(y_true, y_pred):
    """
    决定系数 R²（R-squared），回归任务最常用的"拟合优度"指标。

        R² = 1 - SS_res / SS_tot
        其中 SS_res = Σ (y_i - y_hat_i)^2     （残差平方和：模型没解释掉的部分）
             SS_tot = Σ (y_i - ȳ)^2           （总平方和：数据本身的波动）

    R² 的含义：
        R² = 1  → 完美预测；
        R² = 0  → 模型效果和"直接拿平均值当预测"一样烂；
        R² < 0  → 比直接猜平均值还差（说明模型完全没学到东西）。

    注意：sklearn 里有一模一样的函数，但我们这里自己手写，因为本任务要求"从零实现"。

    参数:
        y_true : 真实值，形状 (n, 1)
        y_pred : 预测值，形状 (n, 1)

    返回:
        r2 : 一个 float
    """

    # 真实值的均值 ȳ。
    y_mean = y_true.mean()

    # 残差平方和：预测错了多少（越小越好）。
    ss_res = ((y_true - y_pred) ** 2).sum()

    # 总平方和：数据自身的方差总量。
    ss_tot = ((y_true - y_mean) ** 2).sum()

    # 防御性编程：如果所有真实值都一样，ss_tot = 0，除法会出错。
    # 这时 R² 没有意义，约定返回 0.0。
    if ss_tot == 0.0:
        return 0.0

    # 按公式计算并返回。
    return float(1.0 - ss_res / ss_tot)


def rmse(y_true, y_pred):
    """
    均方根误差 RMSE（Root Mean Squared Error）。

        RMSE = sqrt( (1/n) * Σ (y_i - y_hat_i)^2 )

    它的好处是"单位"和原始数据一样（这里就是"分"），
    所以可以直接说"平均预测误差大约 5 分"，比 MSE 更好解释。
    """

    # 先用 MSE，再开平方。
    # np.sqrt 可以作用在标量上，所以这里直接调用没问题。
    return float(np.sqrt(mse_loss(y_pred, y_true)))


def mae_metric(y_true, y_pred):
    """
    平均绝对误差 MAE（Mean Absolute Error），单位同样和数据一致。

        MAE = (1/n) * Σ |y_i - y_hat_i|
    """

    # 绝对误差求平均。
    return float(np.abs(y_pred - y_true).mean())


def evaluate_all_metrics(y_true, y_pred):
    """
    一次性算出三个指标，返回一个字典。这样调用处只写一行。
    """

    # 字典的键就是指标名，值是指标数值（都转成 float，方便存 JSON）。
    return {
        "mse": float(mse_loss(y_pred, y_true)),      # 均方误差
        "rmse": rmse(y_true, y_pred),                # 均方根误差
        "mae": mae_metric(y_true, y_pred),           # 平均绝对误差
        "r2": r2_score(y_true, y_pred),              # 决定系数
    }


# ---------------------------- 3. 梯度下降优化算法 ----------------------------

def gradient_descent(model, x_train, y_train, x_test, y_test,
                     learning_rate=0.1, epochs=100, batch_size=None,
                     loss_name="mse", momentum=0.0, lr_decay=1.0,
                     l2=0.0, seed=42, verbose=False, record_test=True):
    """
    用梯度下降训练线性回归模型。这是整个项目的核心函数。

    ============================ 算法原理 ============================
    梯度下降的思想非常朴素：
        损失函数 L(w, b) 像一座山，我们站在山坡上的某一点，
        想走到山谷最低处。虽然看不见全局，但能感觉到脚下的坡度（梯度）。
        于是我们就朝着"下坡最陡"的方向（负梯度方向）迈一小步：

            w := w - learning_rate * dw
            b := b - learning_rate * db

        反复迈步，最终就会走到（凸函数的）最低点。

    其中 learning_rate（学习率）就是"每一步迈多大"：
        太小 → 走得慢，要迭代很多次才收敛；
        太大 → 一步迈过山谷，甚至越走越高，loss 会震荡乃至发散（变成 nan）。

    ============================ 三种变体 ============================
    本函数通过 batch_size 一个参数，就统一实现了三种梯度下降：

        batch_size = None 或 >= 样本总数  →  批量梯度下降 BGD（Batch GD）
            每一步都用【全部 100 个样本】算梯度，方向最准，但每步最贵。

        batch_size = 1                   →  随机梯度下降 SGD（Stochastic GD）
            每一步只用【1 个样本】算梯度，方向很抖，但每步极便宜，
            一个 epoch 里能更新 100 次，通常前期下降飞快。

        batch_size = 中间值（如 16、32）  →  小批量梯度下降 MBGD（Mini-batch GD）
            折中方案：既利用了矩阵运算的并行效率，梯度又比 SGD 稳，
            是深度学习里实际最常用的方式。

    ============================ 两个加速技巧 ============================
    1) 动量（Momentum）：把更新量想象成"小球滚下山"的速度，带惯性。

           v := momentum * v + (1 - momentum) * grad
           w := w - learning_rate * v

       好处：在"峡谷"地形里，来回震荡的方向会互相抵消，
             而持续一致的下坡方向会不断累积，从而加速收敛。

    2) 学习率衰减（Learning Rate Decay）：越接近谷底，步子迈得越小。

           lr_epoch = learning_rate * lr_decay^(epoch-1)

       好处：前期大步快跑，后期小步精调，最终收敛得更稳、更接近最优点。

    ============================ 正则化（可选） ============================
    L2 正则化：在损失里加一项 (l2/2) * w^2，惩罚过大的权重。
        对应的梯度追加一项 l2 * w。
        作用是让模型更"保守"，缓解过拟合。本任务数据很简单，
        主要是为了演示它"有用但不必须"。

    参数:
        model          : LinearRegression 对象（会被就地修改）
        x_train, y_train : 训练集（形状 (n,1)）
        x_test, y_test   : 测试集。只用于【记录】测试 loss 画曲线，
                           不参与任何参数更新（这一点非常重要，否则就是数据泄露）。
        learning_rate  : 学习率（步长）
        epochs         : 训练轮数。1 个 epoch = 把训练集完整过一遍
        batch_size     : 每个批量的样本数；None 表示用全量（BGD）
        loss_name      : "mse" / "mae" / "huber"
        momentum       : 动量系数（0 表示不用动量）
        lr_decay       : 学习率衰减系数（1.0 表示不衰减）
        l2             : L2 正则化强度（0 表示不正则化）
        seed           : 随机种子（打乱顺序、初始化随机状态用）
        verbose        : 是否每 10 个 epoch 打印一次进度
        record_test    : 是否记录测试集 loss（有些实验不需要，可关掉省时间）

    返回:
        history : 字典，包含训练过程的完整记录，用于画图和统计。
    """

    # 取出训练样本个数，后面决定默认 batch_size 时会用到。
    n_train = x_train.shape[0]

    # 如果调用者没有指定 batch_size，就用"全量"，也就是批量梯度下降 BGD。
    if batch_size is None or batch_size >= n_train:
        batch_size = n_train

    # 从损失函数登记表里取出这一次要用的"损失函数"和"梯度函数"。
    loss_fn, grad_fn = LOSS_REGISTRY[loss_name]

    # 准备记录本（都用 Python 列表，最后再转成 numpy 数组方便画图）：
    history = {
        "step": [],               # 每一步（一次参数更新）的编号
        "step_loss": [],          # 每一步在"当前小批量"上的 loss（画出来会比较抖）
        "epoch": [],              # 每一个 epoch 的编号
        "train_loss": [],         # 每个 epoch 结束时，在【整个训练集】上的 MSE
        "test_loss": [],          # 每个 epoch 结束时，在【整个测试集】上的 MSE（只观察）
        "w": [],                  # 每个 epoch 结束时的 w（标准化空间）
        "b": [],                  # 每个 epoch 结束时的 b（标准化空间）
        "grad_norm": [],          # 每个 epoch 第一步的梯度范数（用来观察梯度大小）
        "lr_used": [],            # 每个 epoch 实际使用的学习率（用来看衰减效果）
    }

    # 动量需要一个"速度"变量 v，形状和参数一样，初始为 0。
    # 我们给 w 和 b 各准备一个。
    velocity_w = np.zeros_like(model.w)      # np.zeros_like 建一个"形状相同、全 0"的数组
    velocity_b = np.zeros_like(model.b)

    # 本脚本自己的随机数生成器，保证打乱顺序的过程可复现。
    rng = np.random.default_rng(seed)

    # 记录"参数初始化之后"在训练集上的 loss，作为曲线起点。
    initial_train_loss = float(loss_fn(model.predict(x_train), y_train))

    # 全局步数计数器（每次更新参数 +1）。
    global_step = 0

    # ======================= 关于"溢出警告"的说明 =======================
    # 当我们故意用"过大的学习率"做实验时，参数会指数级增大，
    # 最终超出浮点数能表示的范围，NumPy 会打印一堆 RuntimeWarning
    # （overflow encountered in square / invalid value encountered ...）。
    # 这不是 bug，而正是我们想观察的现象。但这些警告刷屏会盖住真正的信息，
    # 所以这里用 np.errstate 临时把这类警告关掉，训练结束后自动恢复。
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):

        # ======================= 主循环：逐个 epoch =======================
        for epoch in range(1, epochs + 1):

            # ---------- 3.1 计算这个 epoch 实际使用的学习率 ----------
            # lr_decay=1.0 时这一项恒等于 learning_rate，也就是"不衰减"。
            lr_now = learning_rate * (lr_decay ** (epoch - 1))

            # 记录一下，之后可以画"学习率随 epoch 变化"的曲线。
            history["lr_used"].append(lr_now)

            # 本 epoch 里所有小批量 loss 的累加器（用来算平均）。
            epoch_batch_loss_sum = 0.0     # 累加 loss
            epoch_batch_count = 0          # 累加批量个数

            # ---------- 3.2 逐个 epoch 遍历所有小批量 ----------
            # iterate_minibatches 是生成器，每个 epoch 都会重新打乱一次顺序。
            for xb, yb in iterate_minibatches(x_train, y_train, batch_size,
                                              shuffle=True, rng=rng):

                # 前向传播：用当前参数算出这一批的预测值。
                y_hat = model.predict(xb)

                # 计算这一批上的损失（只用来记录，训练本身靠的是梯度）。
                batch_loss = float(loss_fn(y_hat, yb))

                # 计算梯度。grad_fn 会返回 (dw, db)。
                dw, db = grad_fn(y_hat, yb, xb)

                # ---------- 3.3（可选）给梯度加 L2 正则项 ----------
                if l2 > 0.0:
                    # L2 项 (l2/2)*w^2 对 w 的导数是 l2*w；
                    # 对 b 通常不正则化（截距反映数据的整体水平，不该被惩罚）。
                    dw = dw + l2 * model.w

                # 记录梯度范数（size，也就是把数组展平后元素个数）。
                # 这里只在每个 epoch 的第一次更新时记录，避免数据太多。
                if epoch_batch_count == 0:
                    # np.linalg.norm 求向量的欧几里得长度 sqrt(Σx²)。
                    history["grad_norm"].append(float(np.linalg.norm(dw)))

                # ---------- 3.4 用梯度更新参数（梯度下降的核心两行） ----------
                if momentum > 0.0:
                    # 带动量：先更新"速度"，再用速度更新参数。
                    velocity_w = momentum * velocity_w + (1.0 - momentum) * dw
                    velocity_b = momentum * velocity_b + (1.0 - momentum) * db

                    # 用速度代替原始梯度。
                    update_w = velocity_w
                    update_b = velocity_b
                else:
                    # 普通梯度下降：直接拿梯度当更新方向。
                    update_w = dw
                    update_b = db

                # 真正的更新公式：参数 = 参数 - 学习率 × 更新方向
                # 注意中间的减号！梯度指向"上升最快"的方向，
                # 我们要下山，所以要往【反】方向走。
                model.w = model.w - lr_now * update_w
                model.b = model.b - lr_now * update_b

                # 累加这一批的 loss，供 epoch 结束时求平均。
                epoch_batch_loss_sum = epoch_batch_loss_sum + batch_loss
                epoch_batch_count = epoch_batch_count + 1

                # 记录"步级别"的 loss 曲线（会看到明显的抖动，这很正常）。
                global_step = global_step + 1
                history["step"].append(global_step)
                history["step_loss"].append(batch_loss)

            # ---------- 3.5 epoch 结束，做一次"体检" ----------
            # 在整个训练集上评估（注意：这里是评估，不更新参数）。
            train_loss = float(loss_fn(model.predict(x_train), y_train))

            # 记录 epoch 级别的结果。
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)

            # 测试集 loss 只用于"观察泛化情况"，绝不参与训练。
            if record_test:
                history["test_loss"].append(float(loss_fn(model.predict(x_test), y_test)))

            # 记录当前参数，之后可以画出"拟合直线是怎么一步步变好的"。
            history["w"].append(float(model.w.item()))
            history["b"].append(float(model.b.item()))

            # 每 10 轮打印一次进度，方便观察训练有没有正常进行。
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print("    epoch %4d | lr = %.5f | train MSE = %10.4f | test MSE = %10.4f"
                      % (epoch, lr_now, train_loss,
                         history["test_loss"][-1] if record_test else float("nan")))

    # 把函数开头记的"初始 loss"也放进 history，画图时会画成第 0 个点。
    history["initial_train_loss"] = initial_train_loss

    # 把列表转成 numpy 数组，方便后面做切片和数学运算。
    for key in ["step", "step_loss", "epoch", "train_loss", "test_loss",
                "w", "b", "grad_norm", "lr_used"]:
        history[key] = np.array(history[key], dtype=np.float64)

    # 返回完整记录。
    return history


# ---------------------------- 4. 画图小工具 ----------------------------

def clip_curve(y, cap):
    """
    把曲线上超过 cap 的数值"截断"掉，并把被截断的位置标成 nan（空值）。

    为什么需要这个函数？
        当学习率过大时，loss 会以指数速度炸到 1e60 这种天文数字。
        如果原样画在图上，纵轴会被这一个点撑到 1e60，
        其他所有正常曲线会被压成贴着 x 轴的一条直线，什么都看不见。
        所以我们把超过 cap 的部分直接删掉（nan 在 matplotlib 里就是"断开不画"），
        这样图上只剩下"从起点往上冲"的那一小段，既能看到它发散了，又不破坏整张图。

    参数:
        y   : 一条曲线的数值数组
        cap : 上限。超过它的点会被替换成 nan

    返回:
        处理后的新数组（原来的 y 不会被修改）
    """

    # 先复制一份，避免改坏调用者的数据。
    y_new = y.copy()

    # 造一个布尔掩码：哪些点"太大了"。
    too_big = y_new > cap

    # 把这些点设成 nan。nan 是 "not a number"，matplotlib 遇到它会自动断开曲线。
    y_new[too_big] = np.nan

    # 返回处理结果。
    return y_new


def has_diverged(y, cap):
    """
    判断一条曲线的最终值有没有超过 cap，用来在图例里标注"（已发散）"。
    """

    # 取最后一个有效值（过滤掉 nan 之后取最后一个）。
    finite = y[np.isfinite(y)]

    # 如果一个有效值都没有，那就当作发散了。
    if finite.size == 0:
        return True

    # 判断最后一个有效值是否超过上限。
    return bool(finite[-1] > cap)


def save_figure(fig, filename):
    """
    统一保存图片：自动加边距、自动建目录、自动关闭图形释放内存。
    把这段重复逻辑抽成一个函数，是"不要重复自己（DRY）"原则的体现。

    如果全局开关 SAVE_FIGURES 是 False（命令行加了 --no-figures），
    就只关闭图形、不写文件。
    """

    # 自动调整子图间距，避免标签被裁掉。
    fig.tight_layout()

    # 拼出完整输出路径，例如 results/01_loss_curve.png
    out_path = os.path.join(RESULTS_DIR, filename)

    # 判断是否需要真的写文件。
    if SAVE_FIGURES:
        # 保存图片。bbox_inches="tight" 保证外面的文字不会被切掉。
        fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")

    # 关闭图形，释放内存（画很多图时不关会越占越多）。
    plt.close(fig)

    # 打印一下，方便知道文件写到哪了。
    if SAVE_FIGURES:
        print("  [图片] 已保存:", os.path.relpath(out_path, _PROJECT_ROOT))


def write_csv(filename, header, rows):
    """
    手写一个小函数把结果存成 CSV。
    为什么不用 pandas？因为本项目就是想练习"从零实现"，
    而且只是写个表格，Python 自带的 csv 模块完全够用。
    """

    # 拼出完整路径。
    out_path = os.path.join(RESULTS_DIR, filename)

    # newline="" 是 csv 模块在 Windows 上的推荐写法，避免每行多出一个空行。
    # encoding="utf-8-sig" 会在文件开头写一个 BOM，这样用 Excel 打开中文不会乱码。
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:

        # csv.writer(f) 创建一个"写入器"对象，用它才能正确地处理逗号、引号等转义。
        writer = csv.writer(f)

        # 先写表头（第一行，说明每一列是什么）。
        writer.writerow(header)

        # 再逐行写数据。
        for row in rows:
            writer.writerow(row)

    # 提示保存成功。
    print("  [数据] 已保存:", os.path.relpath(out_path, _PROJECT_ROOT))


# ---------------------------- 5. 结果汇总小工具 ----------------------------

def json_safe(value):
    """
    把 numpy 里的数字（np.float64、np.int64）转成普通的 Python 数字，
    把 nan / inf 转成字符串，这样 json.dump 才不会报错。

    为什么需要它？
        - json 模块不认识 numpy 的标量类型，直接存会抛 TypeError；
        - json 标准里也没有 NaN 和 Infinity，写进去的文件别的语言读不了。
        这两个坑在做"保存实验结果"这件事时几乎一定会遇到。
    """

    # np.floating 包含 np.float64、np.float32 等所有 numpy 浮点类型。
    if isinstance(value, np.floating):
        # 先转成 Python 的 float。
        value = float(value)

    # np.integer 包含 np.int64、np.int32 等所有 numpy 整数类型。
    if isinstance(value, np.integer):
        # 转成 Python 的 int。
        value = int(value)

    # 如果现在是一个 float，检查它是不是 nan 或 inf。
    if isinstance(value, float):
        # np.isfinite 为 False 就说明是 nan 或 ±inf。
        if not np.isfinite(value):
            # 换成字符串，保证 JSON 合法。
            return "nan" if value != value else ("inf" if value > 0 else "-inf")

    # 其余类型原样返回。
    return value


def to_json_safe(obj):
    """
    递归地把整个字典/列表里的 numpy 数字都转换成 JSON 能接受的类型。

    这叫"递归"：遇到字典就处理它的每个值，遇到列表就处理它的每个元素，
    直到遇到最底层的普通数字为止。
    """

    # 如果是字典，就处理它的每一个值。
    if isinstance(obj, dict):
        # 用字典推导式生成一个新字典（不修改原来的）。
        return {key: to_json_safe(val) for key, val in obj.items()}

    # 如果是列表或元组，就处理它的每一个元素。
    if isinstance(obj, (list, tuple)):
        # 统一返回列表。
        return [to_json_safe(val) for val in obj]

    # 到了最底层，就用 json_safe 处理单个值。
    return json_safe(obj)


def summarize_reach_epoch(history, target_loss):
    """
    计算"这条训练曲线第几轮进入目标 loss 以内"。

    这是评价优化算法好坏的关键指标：
    只看最后一轮的 loss 容易被随机波动误导，而"多久到达最优附近"更能反映速度。

    参数:
        history     : gradient_descent 返回的历史记录（里面有 "train_loss"）
        target_loss : 目标损失值

    返回:
        第几轮达标（int）；如果从头到尾都没达标，返回 None
    """

    # 取出训练 loss 曲线。
    curve = history["train_loss"]

    # np.where 找出所有 <= target_loss 的位置下标。
    hit = np.where(curve <= target_loss)[0]

    # 有就返回"下标 + 1"（因为下标从 0 开始，轮数从 1 开始）；没有就返回 None。
    return (int(hit[0]) + 1) if hit.size > 0 else None


def momentum_final_and_reach(momentum_results, target_loss):
    """
    从实验 F 的结果里，提取"每个场景、每个配置的最终 loss 和达标轮数"。

    返回的东西会直接写进 results.json，这样生成报告时就不用把数字抄一遍，
    报告里的每一个数字都能追溯到真实的训练结果。
    """

    # 准备一个空字典装结果。
    table = {}

    # 遍历三个场景。
    for scene_name in momentum_results:

        # 每个场景又是一个小字典。
        table[scene_name] = {}

        # 遍历这个场景里的四个配置。
        for cfg_name in momentum_results[scene_name]:

            # 取出这个配置的训练历史。
            history = momentum_results[scene_name][cfg_name]

            # 记下最终 loss 和达标轮数。
            table[scene_name][cfg_name] = {
                "final_train_mse": float(history["train_loss"][-1]),
                "reach_epoch": summarize_reach_epoch(history, target_loss),
            }

    # 返回整理好的表格。
    return table


# ---------------------------- 6. 实验部分 ----------------------------

def experiment_baseline(data, rng):
    """
    实验 0（基线）：用闭式解（正规方程）算出理论最优参数。

    这个实验本身不训练模型，它的作用是给我们一个"标准答案"：
    后面梯度下降训练出来的参数，应该和它非常接近。
    如果不接近，就说明梯度下降还没收敛，或者代码有 bug。
    """

    print("\n[实验 0] 基线：闭式解（正规方程）—— 给梯度下降当参照物")

    # 在【标准化后】的训练数据上求闭式解。
    # 用标准化数据是为了和梯度下降在同一个坐标系里比较。
    w_std, b_std = fit_closed_form(data["x_train_std"], data["y_train"])

    # 把标准化空间的参数换算回原始空间（hours 单位），这样人能看懂。
    w_raw, b_raw = restore_params(w_std, b_std, data["mean"], data["std"])

    # 打印结果。
    print("  标准化空间: w = %.6f, b = %.6f" % (w_std, b_std))
    print("  原始空间:   score = %.4f * hours + %.4f" % (w_raw, b_raw))

    # 顺便用闭式解在测试集上评估一次，作为"最优水平"的参考。
    model_cf = LinearRegression()                 # 新建模型
    model_cf.set_params(w_std, b_std)             # 装入闭式解
    pred_test = model_cf.predict(data["x_test_std"])   # 在测试集上预测
    metrics = evaluate_all_metrics(data["y_test"], pred_test)   # 算指标

    # 打印测试集指标。
    print("  闭式解在测试集上: MSE=%.4f RMSE=%.4f MAE=%.4f R2=%.4f"
          % (metrics["mse"], metrics["rmse"], metrics["mae"], metrics["r2"]))

    # 返回结果字典，供后续使用和保存。
    return {
        "w_std": float(w_std), "b_std": float(b_std),      # 标准化空间参数
        "w_raw": float(w_raw), "b_raw": float(b_raw),      # 原始空间参数
        "test_metrics": metrics,                            # 测试集指标
    }


def experiment_loss_curve(data):
    """
    实验 A：最基础的收敛实验。
    用批量梯度下降（BGD）+ 一个合适的学习率，观察：
        1) loss 是不是稳定下降？
        2) 拟合直线是怎么一步步贴近真实数据点的？

    注意：本函数同时负责"训练"和"画图"。
    如果只是想让训练跑完、不需要画图（比如只想快速验证代码），
    可以在运行时加 --no-figures 开关；训练本身仍然会正常执行。
    """

    print("\n[实验 A] Loss 下降曲线 + 拟合直线随迭代的演化")

    # 新建模型（w=0, b=0）。
    model = LinearRegression()

    # 这里故意用"批量梯度下降"（batch_size 不传 → 全量），因为它的曲线最干净，
    # 适合用来讲解"梯度下降是怎么下降的"。
    history = gradient_descent(
        model,
        data["x_train_std"], data["y_train"],
        data["x_test_std"], data["y_test"],
        learning_rate=0.1,        # 学习率
        epochs=200,               # 迭代 200 轮
        batch_size=None,          # None → 全量，即批量梯度下降 BGD
        loss_name="mse",          # 用均方误差
        seed=42,
    )

    # 训练结束后，把参数换算回原始空间，看看学到的直线长什么样。
    w_std, b_std = model.get_params()
    w_raw, b_raw = restore_params(w_std, b_std, data["mean"], data["std"])
    print("  训练完成: score = %.4f * hours + %.4f" % (w_raw, b_raw))
    print("  最终训练 MSE = %.4f" % history["train_loss"][-1])

    # ---------------- 画图 1：loss 曲线（线性坐标 + 对数坐标） ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))   # 1 行 2 列的子图

    # 左图：普通坐标。x 是 epoch，y 是 MSE。
    axes[0].plot(history["epoch"], history["train_loss"],
                 color=C_TRAIN, linewidth=2, label="训练集 MSE")
    axes[0].plot(history["epoch"], history["test_loss"],
                 color=C_TEST, linewidth=2, linestyle="--", label="测试集 MSE")
    axes[0].set_xlabel("迭代轮数 epoch")
    axes[0].set_ylabel("损失 Loss (MSE)")
    axes[0].set_title("(a) Loss 随迭代次数的下降曲线")
    axes[0].legend()          # 显示图例
    axes[0].grid(alpha=0.3)   # 画浅色网格，方便读数

    # 右图：对数坐标。因为 loss 从 3000 降到 50，直接用线性坐标
    # 后面那段下降过程会被压成一条贴地的线，看不出细节。
    # 换成 log 坐标后，整个下降过程就都能看清了。
    axes[1].plot(history["epoch"], history["train_loss"],
                 color=C_TRAIN, linewidth=2, label="训练集 MSE")
    axes[1].plot(history["epoch"], history["test_loss"],
                 color=C_TEST, linewidth=2, linestyle="--", label="测试集 MSE")
    axes[1].set_yscale("log")     # 把 y 轴改成对数刻度
    axes[1].set_xlabel("迭代轮数 epoch")
    axes[1].set_ylabel("损失 Loss (MSE, 对数刻度)")
    axes[1].set_title("(b) 同样的曲线，纵轴取对数后细节更清楚")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")   # 对数坐标下主次网格都要画

    # 保存这张图。
    save_figure(fig, "01_loss_curve.png")

    # ---------------- 画图 2：拟合直线随迭代的演化 ----------------
    # 我们挑 6 个有代表性的 epoch，把当时的直线画在同一张图上，
    # 直观展示"从一条烂线，慢慢转成一条好线"的过程。
    show_epochs = [1, 2, 5, 20, 60, 200]          # 要展示的 epoch 编号

    fig, ax = plt.subplots(figsize=(8.6, 6.0))

    # 先画原始数据点（散点图），作为背景。
    ax.scatter(data["x_train_raw"].reshape(-1), data["y_train"].reshape(-1),
               s=26, color=C_TRAIN, alpha=0.65, label="训练集数据点", zorder=3)

    # 准备一条"横坐标网格"，用来画直线：从最小 hours 到最大 hours 取 100 个点。
    x_grid_raw = np.linspace(data["x_train_raw"].min(), data["x_train_raw"].max(), 100)

    # 线性插值出对应的标准化横坐标，这样可以用模型直接预测。
    x_grid_std = to_column((x_grid_raw - data["mean"]) / data["std"])

    # 用渐变的颜色，表示"越晚的 epoch 颜色越深"。
    # np.linspace 生成 0.25 → 0.95 的一组透明度。
    alphas = np.linspace(0.25, 0.95, len(show_epochs))

    # 逐个 epoch 画一条线。
    for i, ep in enumerate(show_epochs):
        # history 里的数组下标从 0 开始，所以第 ep 轮对应下标 ep-1。
        w_ep = history["w"][ep - 1]
        b_ep = history["b"][ep - 1]

        # 预测这条线上所有点的 y 值。
        y_grid = (x_grid_std @ np.array([[w_ep]])) + b_ep

        # 画线。ymax 就是预测值，reshape(-1) 把它变回一维方便画图。
        ax.plot(x_grid_raw, y_grid.reshape(-1),
                color=C_LINE, alpha=alphas[i], linewidth=2.0,
                label="epoch %d" % ep, zorder=2)

    # 也画一下闭式解（理论最优）那条线，作为"目标"参考。
    w_cf, b_cf = fit_closed_form(data["x_train_std"], data["y_train"])
    w_cf_raw, b_cf_raw = restore_params(w_cf, b_cf, data["mean"], data["std"])
    ax.plot(x_grid_raw, w_cf_raw * x_grid_raw + b_cf_raw,
            color="black", linewidth=1.4, linestyle=":", label="闭式解（理论最优）", zorder=4)

    # 设置坐标轴标签和标题。
    ax.set_xlabel("学习时长 hours")
    ax.set_ylabel("成绩 score")
    ax.set_title("拟合直线随迭代次数的演化（批量梯度下降，lr=0.1）")

    # 图例放在右下角，避免挡住数据。
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # 保存这张图。
    save_figure(fig, "02_fit_progress.png")

    # 返回这个实验的 history，后面可以复用。
    return history


def experiment_learning_rate(data, lr_list):
    """
    实验 B：学习率对收敛的影响 —— 本任务最重要的实验之一。

    我们要亲眼看到三件事：
        1) 学习率太小 → 收敛慢，200 轮还没走到最低点；
        2) 学习率合适 → 快速平滑地收敛；
        3) 学习率太大 → 震荡甚至发散（loss 变成 nan / inf）。
    """

    print("\n[实验 B] 不同学习率的收敛情况对比")

    # 用字典保存每个学习率的训练历史，键就是学习率。
    histories = {}

    # 逐个学习率单独训练一次。
    for lr in lr_list:
        print("  --- 学习率 lr = %g ---" % lr)

        # 每次都用全新的模型，避免上一次的训练结果"污染"这一次。
        model = LinearRegression()

        # 全部使用批量梯度下降（BGD），这样唯一的变量就是学习率，实验才干净。
        history = gradient_descent(
            model,
            data["x_train_std"], data["y_train"],
            data["x_test_std"], data["y_test"],
            learning_rate=lr,
            epochs=200,
            batch_size=None,          # 全量，BGD
            loss_name="mse",
            seed=42,
        )

        # 把结果存起来。
        histories[lr] = history

        # 打印最后几轮的 loss，看它到底是收敛了还是发散了。
        final_loss = history["train_loss"][-1]
        print("      最终 train MSE = %.4f" % final_loss)

    # ---------------- 画图 3：不同学习率的 loss 曲线 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # 为了让每条线颜色不同，用一个颜色表按顺序取色。
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(lr_list)))

    # 纵轴的显示上限。loss 初始值约 3000，所以 1e5 足够看清正常曲线，
    # 又能让"发散的曲线"在冲出这个上限时被截断，不破坏整张图。
    y_cap = 1e5

    # 左图：普通坐标。发散的曲线会冲上去，正好用来展示"学习率过大"。
    for i, lr in enumerate(lr_list):
        # 把超过上限的部分截断，避免纵轴被 1e60 这种数字撑爆。
        y_plot = clip_curve(histories[lr]["train_loss"], y_cap)

        # 在图例里标注哪些学习率已经发散。
        if has_diverged(histories[lr]["train_loss"], y_cap):
            label = "lr = %g（已发散）" % lr
        else:
            label = "lr = %g" % lr

        axes[0].plot(histories[lr]["epoch"], y_plot,
                     color=colors[i], linewidth=1.8, label=label)

    # 画出显示上限的位置，并解释"冲出画面的线"意味着什么。
    axes[0].axhline(y_cap, color=C_GRAY, linestyle=":", linewidth=1.2)
    axes[0].annotate("此线以上已超出显示范围\n（说明该学习率下的 loss 正在爆炸）",
                     xy=(0.03, 0.80), xycoords="axes fraction", fontsize=8.5,
                     color=C_GRAY)

    axes[0].set_xlabel("迭代轮数 epoch")
    axes[0].set_ylabel("训练集 Loss (MSE)")
    axes[0].set_title("(a) 线性纵轴：学习率过大时 loss 直接爆炸")
    axes[0].legend(fontsize=8.5, loc="center right")
    axes[0].grid(alpha=0.3)

    # 右图：对数坐标，只看收敛区间，方便比较"谁下降得更快"。
    for i, lr in enumerate(lr_list):
        # 同样做截断处理，保证对数轴的范围合理。
        y_plot = clip_curve(histories[lr]["train_loss"], y_cap)

        axes[1].plot(histories[lr]["epoch"], y_plot,
                     color=colors[i], linewidth=1.8, label="lr = %g" % lr)

    # 用一条水平虚线标出"理论最优 loss"，让大家看到收敛的曲线都压在这条线上。
    w_cf, b_cf = fit_closed_form(data["x_train_std"], data["y_train"])
    model_cf = LinearRegression()
    model_cf.set_params(w_cf, b_cf)
    best_possible = float(mse_loss(model_cf.predict(data["x_train_std"]), data["y_train"]))
    axes[1].axhline(best_possible, color="black", linestyle="--", linewidth=1.3,
                    label="理论最优 %.1f（闭式解）" % best_possible)

    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=10.0, top=y_cap)     # 手动限定范围，图更好看
    axes[1].set_xlabel("迭代轮数 epoch")
    axes[1].set_ylabel("训练集 Loss (MSE, 对数刻度)")
    axes[1].set_title("(b) 对数纵轴：小学习率下降慢，合适的收敛快")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3, which="both")

    # 保存。
    save_figure(fig, "03_learning_rate.png")

    # ---------------- 画图 4：学习率 vs 最终 loss（"U 形"曲线） ----------------
    # 这张图想说明一个关键结论：学习率不是越大越好，也不是越小越好，
    # 中间有一个"甜蜜点（sweet spot）"，太大的话反而会发散。
    final_losses = []       # 每个学习率最终的 loss
    for lr in lr_list:
        # 取最后一轮的 loss；如果是 nan（发散），这里保持 nan。
        final_losses.append(histories[lr]["train_loss"][-1])

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    # 画折线 + 数据点。
    # 这里也要截断：否则一个 1e62 的点会让纵轴变成 1e62，
    # 其余所有点就都被压到最底下、看不出差别了。
    y_plot = clip_curve(np.array(final_losses, dtype=np.float64), y_cap)

    ax.plot(lr_list, y_plot, marker="o", color=C_TRAIN, linewidth=2,
            markersize=7, label="最终训练 Loss（超过 %.0e 的部分不显示）" % y_cap)

    # 用对数横轴，因为学习率是从 0.001 到 1.2 的大范围变化。
    ax.set_xscale("log")

    # 纵轴也用对数，否则在 1 ~ 2000 之间的小差别看不出来。
    ax.set_yscale("log")

    # 手动限制纵轴范围，让"正常收敛"的那几个点也能看清。
    ax.set_ylim(bottom=10.0, top=y_cap * 3.0)

    # 标注每个点对应的数值，方便读图。
    for lr, fl in zip(lr_list, final_losses):
        # 如果 loss 超出了显示范围（发散），文字就写"发散"，并把点画在显示上限处，
        # 这样读者能明确看到"是哪个学习率冲出了画面"。
        if not np.isfinite(fl) or fl > y_cap:
            text = "lr=%g 发散出界" % lr
            y_point = y_cap * 1.5          # 把点放在上部，表示"它已经冲出画面了"
            ax.plot([lr], [y_point], marker="x", color=C_TEST, markersize=10,
                    markeredgewidth=2.5, zorder=5)
        else:
            text = "%.1f" % fl
            y_point = fl                   # 正常值就标在点上方

        # ha 控制水平对齐，xytext 把文字往上挪一点避免压住点。
        ax.annotate(text, (lr, y_point), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5)

    # 用一条水平虚线标出"闭式解的理论最优 loss"，作为参照线。
    ax.axhline(best_possible, color=C_GRAY, linestyle="--", linewidth=1.4,
               label="理论最优 (闭式解) = %.1f" % best_possible)

    ax.set_xlabel("学习率 learning rate (对数刻度)")
    ax.set_ylabel("200 轮后的训练集 Loss (对数刻度)")
    ax.set_title("学习率对最终收敛效果的影响（MSE 在 lr 过大时发散）")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    # 保存。
    save_figure(fig, "04_lr_vs_final_loss.png")

    # 返回所有历史，供结果保存使用。
    return histories, final_losses


def experiment_standardization(data, histories_std, lr_list):
    """
    实验 C：特征标准化 vs 不标准化。

    这个实验回答一个很常见的问题：
        "我用原始数据训练，loss 变成 nan 了，是不是代码写错了？"
    答案是：往往不是代码错，而是【特征尺度不匹配 + 学习率太大】。

    原理：对线性回归 y = w*x + b，MSE 的 Hessian（二阶导矩阵）里，
          w 方向的曲率正比于 Σx²，b 方向的曲率正比于 n。
          如果 x 的量级是 10，Σx²/n ≈ 100，而 b 方向只有 1，
          两个方向的曲率差了 100 倍，等高线是被拉得极细长的椭圆。
          这时，能让长轴方向不发散的学习率，对短轴方向就太小（收敛慢）；
          而能让短轴方向快的学习率，长轴方向就会震荡发散。
          更严格地说，梯度下降稳定的条件是 lr < 2 / L，
          其中 L 是 Hessian 的最大特征值。标准化后 L 大幅减小，可用的 lr 范围就宽多了。
    """

    print("\n[实验 C] 特征标准化 vs 原始尺度（都跑一遍同样的学习率）")

    # 原始（未标准化）的训练特征，注意形状是 (n, 1)。
    x_train_raw = data["x_train_raw"]
    y_train = data["y_train"]

    # 为了公平对比，我们让"原始数据"这一组也用同样的 target 尺度（成绩 30~100）。
    # 用同一个 lr 列表跑一遍。
    histories_raw = {}

    # 逐个学习率训练。
    for lr in lr_list:
        # 新模型。
        model = LinearRegression()

        # 注意：这里传进去的是【原始】x_train_raw，没有做标准化。
        history = gradient_descent(
            model,
            x_train_raw, y_train,                 # ← 关键区别：原始尺度
            x_train_raw, y_train,                 # 测试集也随便传一个（这个实验不看测试集）
            learning_rate=lr,
            epochs=200,
            batch_size=None,                      # BGD
            loss_name="mse",
            seed=42,
            record_test=False,                    # 不需要测试曲线，省点计算
        )

        # 记录。
        histories_raw[lr] = history
        print("  原始尺度 lr=%-6g → 最终 train MSE = %.4f" % (lr, history["train_loss"][-1]))

    # ---------------- 画图 5：两条曲线对比 ----------------
    # 挑两个学习率来画：一个"原始数据会爆炸"的大学习率，一个很小、收敛很慢的学习率。
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

    # 显示上限：原始尺度在大学习率下会冲到 1e306，必须截断，否则图没法看。
    y_cap = 1e6

    # 左图：大学习率（例如 0.3）下，标准化 vs 原始尺度。
    lr_big = lr_list[len(lr_list) // 2]     # 取列表中间那个学习率

    # 原始尺度的曲线（截断处理）。
    y_raw = clip_curve(histories_raw[lr_big]["train_loss"], y_cap)
    axes[0].plot(histories_raw[lr_big]["epoch"], y_raw,
                 color=C_TEST, linewidth=2, label="原始尺度 (未标准化)")

    # 标准化后的曲线。
    axes[0].plot(histories_std[lr_big]["epoch"],
                 clip_curve(histories_std[lr_big]["train_loss"], y_cap),
                 color=C_TRAIN, linewidth=2, label="标准化后 (mean=0, std=1)")

    # 纵轴用对数刻度，这样"一条平稳下降"和"一条垂直爆炸"能同时看清。
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=10.0, top=y_cap * 3.0)
    axes[0].set_xlabel("迭代轮数 epoch")
    axes[0].set_ylabel("训练集 Loss (MSE, 对数刻度)")
    axes[0].set_title("(a) 同一个学习率 lr=%g：原始尺度直接爆炸，标准化后稳稳收敛" % lr_big)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, which="both")

    # 右图：把"可用的学习率范围"直接画出来 —— 这才是标准化最本质的好处。
    # 横轴是学习率，纵轴是 200 轮后的训练 loss，两条线分别是"原始尺度"和"标准化"。
    # 结论会非常直观：原始尺度只要 lr >= 0.05 就爆炸（曲线冲出画面），
    # 而标准化后一直到 lr = 1 都还能稳定收敛 —— 可用的学习率范围宽了几十倍。
    raw_final = []          # 原始尺度在各自学习率下的最终 loss
    std_final = []          # 标准化在各自学习率下的最终 loss

    # 逐一对齐两个列表。
    for lr in lr_list:
        raw_final.append(histories_raw[lr]["train_loss"][-1])
        std_final.append(histories_std[lr]["train_loss"][-1])

    # 转成 numpy 数组，并做截断处理，否则一个 1e306 会把纵轴撑到看不见别的东西。
    raw_final = clip_curve(np.array(raw_final, dtype=np.float64), y_cap)
    std_final = clip_curve(np.array(std_final, dtype=np.float64), y_cap)

    # 画两条折线。
    axes[1].plot(lr_list, raw_final, marker="o", color=C_TEST, linewidth=2,
                 markersize=6, label="原始尺度 (未标准化)")
    axes[1].plot(lr_list, std_final, marker="s", color=C_TRAIN, linewidth=2,
                 markersize=6, label="标准化后 (mean=0, std=1)")

    # 横轴取对数（学习率跨了好几个数量级），纵轴也取对数。
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=10.0, top=y_cap * 3.0)

    axes[1].set_xlabel("学习率 learning rate (对数刻度)")
    axes[1].set_ylabel("200 轮后的训练集 Loss (对数刻度)")
    axes[1].set_title("(b) 标准化把可用的学习率范围拓宽了几十倍")
    axes[1].legend(fontsize=9, loc="upper left")
    axes[1].grid(alpha=0.3, which="both")

    # 保存。
    save_figure(fig, "05_standardization.png")

    # 返回原始尺度的训练历史，供结果统计。
    return histories_raw


def experiment_batch_size(data, configs):
    """
    实验 D：批量大小对训练过程的影响（BGD / Mini-batch / SGD）。

    configs 是一个列表，每个元素是 (显示名字, batch_size, 学习率)。
    给 SGD 单独配一个更小的学习率，因为它单步梯度噪声大，用大学习率会炸。
    """

    print("\n[实验 D] 批量大小对比：BGD / Mini-batch / SGD")

    # 存每个配置的训练历史。
    results = {}

    # 逐个配置训练。
    for name, bs, lr in configs:
        print("  --- %s (batch_size=%s, lr=%g) ---" % (name, bs, lr))

        # 新模型。
        model = LinearRegression()

        # 训练。
        history = gradient_descent(
            model,
            data["x_train_std"], data["y_train"],
            data["x_test_std"], data["y_test"],
            learning_rate=lr,
            epochs=60,                    # 轮数少一点，图比较好看清
            batch_size=bs,                # ← 唯一的自变量
            loss_name="mse",
            seed=42,
        )

        # 记录。
        results[name] = history

        # 打印：注意这里同时看"轮级别"和"步级别"的 loss。
        print("      最终 train MSE = %.4f（共 %d 次参数更新）"
              % (history["train_loss"][-1], int(history["step"][-1])))

    # ---------------- 画图 6：三种视角看批量大小 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # 左图：按【epoch】为横轴，看"整个训练集上的 loss"随轮数的变化。
    # 这里能看出 SGD 下降最快——不是因为它每步更聪明，
    # 而是因为它在同样一轮里更新了 100 次（BGD 只更新 1 次）。
    for name in results:
        axes[0].plot(results[name]["epoch"], results[name]["train_loss"],
                     linewidth=2, label=name)
    axes[0].set_xlabel("迭代轮数 epoch")
    axes[0].set_ylabel("训练集 Loss (MSE)")
    axes[0].set_title("(a) 按 epoch：SGD 每轮更新次数最多，下降最快")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # 中图：按【参数更新次数 step】为横轴，看每一步的 batch loss。
    # 这张图最重要的作用是让我们看到"抖动（噪声）"：
    #   SGD 的曲线剧烈震荡（每步只看 1 个样本，方向很偏）；
    #   BGD 的曲线极其平滑（每步看全部样本，方向很准）；
    #   Mini-batch 居中 —— 这正是它成为深度学习默认选择的原因。
    for name in results:
        # SGD 有 6000 步，全部画出来太密，所以每隔几个点抽一个画（降采样）。
        step = results[name]["step"]
        loss = results[name]["step_loss"]
        stride = max(1, len(step) // 600)      # 最多画 600 个点
        axes[1].plot(step[::stride], loss[::stride],
                     linewidth=0.9, alpha=0.8, label=name)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("参数更新次数 step")
    axes[1].set_ylabel("当前批量的 Loss (MSE, 对数刻度)")
    axes[1].set_title("(b) 按 step：SGD 抖动剧烈，BGD 平滑")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    # 右图：算一笔"性价比"的账。
    # 我们定义两个量：
    #   目标 = 损失降到"理论最优的 1.05 倍"以内（也就是离最优只差 5%）
    #   代价 = 达到目标的过程中，一共让多少个样本数据经过了计算
    #          （这才是真正花时间的部分：每处理一个样本就要算一次预测和梯度）
    #       BGD：        每轮 100 个样本 → 每轮代价 100
    #       Mini-batch： 每轮 100 个样本 → 每轮代价 100（只是分 7 批算，总样本数一样）
    #       SGD：        每轮 100 个样本 → 同样每轮 100
    #   所以"代价"其实就等于 100 × 达到目标的轮数。
    #   最后列出来的柱状图 = "要花多少代价才能达标"，柱子越矮越划算。
    names_panel = list(results.keys())          # 四个配置的名字

    # 每条曲线的 batch_size 是多少（从配置里取回来）。
    batch_of_name = {}
    for name, bs, lr in configs:
        # bs 为 None 表示全量，也就是样本总数。
        batch_of_name[name] = data["n_train"] if bs is None else bs

    # 理想损失（闭式解）。
    w_cf2, b_cf2 = fit_closed_form(data["x_train_std"], data["y_train"])
    model_cf2 = LinearRegression()
    model_cf2.set_params(w_cf2, b_cf2)
    loss_best = float(mse_loss(model_cf2.predict(data["x_train_std"]), data["y_train"]))

    # 目标损失 = 最优的 1.05 倍。
    target_loss = loss_best * 1.05

    # 逐个配置找"第一次达到目标的轮数"。
    reach_epoch = {}       # 名字 → 第几轮达标
    reach_cost = {}        # 名字 → 达标时的累计计算代价
    for name in names_panel:
        # 取出这条曲线的训练 loss 数组。
        train_curve = results[name]["train_loss"]

        # 找到第一个 <= target_loss 的位置下标。
        # np.where 返回满足条件的下标数组，取第 0 个就是"第一次达标"。
        hit = np.where(train_curve <= target_loss)[0]

        if hit.size == 0:
            # 如果 60 轮都没达标（比如 lr 太小的那种），就视作"未达标"，
            # 用总轮数当占位，并在图上标注说明。
            reach_epoch[name] = None
            reach_cost[name] = len(train_curve) * data["n_train"]
        else:
            # 达标了：轮数 = 下标 + 1（下标从 0 开始）。
            ep = int(hit[0]) + 1
            reach_epoch[name] = ep
            reach_cost[name] = ep * data["n_train"]

    # 画柱状图。
    costs = [reach_cost[name] for name in names_panel]
    bars3 = axes[2].bar(range(len(names_panel)), costs,
                        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                        alpha=0.85, width=0.62)

    # 在柱子上标注"第几轮达标 / 达标时更新了多少次"。
    for bar, name in zip(bars3, names_panel):
        ep = reach_epoch[name]
        text_ep = "未达标" if ep is None else "第 %d 轮达标" % ep
        # 更新次数 = 轮数 × 每轮批次数。
        n_updates_at_target = 0 if ep is None else ep * int(np.ceil(data["n_train"] / batch_of_name[name]))
        axes[2].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() * 1.03,
                     "%s\n更新 %d 次" % (text_ep, n_updates_at_target),
                     ha="center", fontsize=8.0)

    # 横轴用配置名，太长了要换成短名字，否则会叠在一起。
    axes[2].set_xticks(range(len(names_panel)))
    axes[2].set_xticklabels(["BGD\nbatch=100\nlr=0.05",
                             "Mini-batch\nbatch=16\nlr=0.05",
                             "SGD\nbatch=1\nlr=0.05",
                             "SGD\nbatch=1\nlr=0.01"], fontsize=7.5)

    # 纵轴留出空间给文字标注。
    axes[2].set_ylim(0, max(costs) * 1.28)
    axes[2].set_ylabel("达标所需的总计算代价\n（处理过的样本数）")
    axes[2].set_title("(c) 谁最省算力？（损失降到最优的 1.05 倍以内）")
    axes[2].grid(alpha=0.3, axis="y")

    # 保存。
    save_figure(fig, "06_batch_size.png")

    # 返回结果。
    return results


def experiment_loss_functions(data, outlier_indices, outlier_shift):
    """
    实验 E：不同损失函数对异常值（outlier）的鲁棒性。

    做法：往训练集里人为塞几个"异常点"——比如某个学生只学了 3 小时却拿了 100 分
    （现实中可能是作弊、或者数据录入错误）。
    然后分别用 MSE、MAE、Huber 训练，看看谁的直线被这几个坏点带偏得最少。

    预期结论：
        MSE 被带偏最多（因为误差被平方，一个离谱的点的"话语权"极大）；
        MAE 最稳（误差只按一次方算，大误差不会获得额外权重）；
        Huber 居中（小误差像 MSE，大误差像 MAE）。

    ⚠️ 这里踩过一个真实的坑（详见 note.md 与报告）：
        一开始我用【批量梯度下降 BGD】来训练 MAE，结果它 200 轮几乎没动，
        直线一直是 y=0。原因不是代码错，而是数学性质：
        MAE 的梯度只有 ±1 这种"固定大小"，不会像 MSE 那样在接近最优点时自动变小，
        而且梯度里乘的是 x（标准化后 |x|<=1.5），所以步长极小，2000 轮都走不到最优。
        解决办法有二：(1) 换用 mini-batch 把梯度尺度放大；(2) 给 MAE 单独用很小的学习率。
        下面我们统一换成【mini-batch，batch=16，lr=0.05】，这样三种损失都能又快又稳地收敛，
        对比才是公平的。
    """

    print("\n[实验 E] 损失函数对比：MSE / MAE / Huber 抗异常值能力")

    # 复制一份训练数据，避免污染全局数据（copy 很重要，不然改的就是原数组）。
    x_train = data["x_train_std"].copy()
    y_train = data["y_train"].copy()

    # 在指定的几个位置上人为制造异常值：把成绩硬加上一个很大的偏移。
    # 例如原来 40 分，变成 85 分。
    for idx in outlier_indices:
        y_train[idx, 0] = y_train[idx, 0] + outlier_shift

    # 打印一下被改动的样本，方便对照。
    print("  已在训练集中注入 %d 个异常点（成绩 +%g 分），位置: %s"
          % (len(outlier_indices), outlier_shift, list(outlier_indices)))

    # 先算一下"没有被污染时"的正确斜率，作为后面衡量"被带偏多少"的基准。
    # 用闭式解（正规方程）算，快速又精确。
    w_clean_std, b_clean_std = fit_closed_form(data["x_train_std"], data["y_train"])
    w_clean_raw, b_clean_raw = restore_params(w_clean_std, b_clean_std,
                                              data["mean"], data["std"])

    # 三种损失函数各训练一次。
    results = {}
    for loss_name in ["mse", "mae", "huber"]:
        print("  --- 损失函数: %s ---" % loss_name)

        # 新模型。
        model = LinearRegression()

        # 统一的优化配置：mini-batch(16) + lr=0.05，300 轮。
        # 三种损失用完全相同的配置，这样"唯一的自变量"就真的只有损失函数本身。
        history = gradient_descent(
            model,
            x_train, y_train,
            data["x_test_std"], data["y_test"],
            learning_rate=0.05,       # 学习率（三种损失相同）
            epochs=300,               # 迭代轮数（三种损失相同）
            batch_size=16,            # 小批量大小（三种损失相同）
            loss_name=loss_name,      # ← 唯一的自变量
            seed=42,
            record_test=False,        # 这个实验只看拟合直线，不需要测试曲线
        )

        # 记录参数（标准化空间）。
        w_std, b_std = model.get_params()

        # 换算到原始空间，方便打印和画图。
        w_raw, b_raw = restore_params(w_std, b_std, data["mean"], data["std"])

        # 量化"被异常值带偏了多少"：斜率与正确斜率的相对偏差（百分比）。
        slope_error_pct = abs(w_raw - w_clean_raw) / abs(w_clean_raw) * 100.0

        # 把这一组的结果打包保存。
        results[loss_name] = {
            "w_std": float(w_std),
            "b_std": float(b_std),
            "w_raw": float(w_raw),
            "b_raw": float(b_raw),
            "slope_error_pct": float(slope_error_pct),
            "final_train_loss": float(history["train_loss"][-1]),
            "history": history,
        }

        print("      学到: score = %.4f * hours + %.4f  |  斜率偏离正确值 %.2f%%"
              % (w_raw, b_raw, slope_error_pct))

    # ---------------- 画图 7：三条拟合直线 + 量化对比 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # 准备一条"横坐标网格"，用来画直线。
    x_grid_raw = np.linspace(data["x_train_raw"].min(), data["x_train_raw"].max(), 100)
    x_grid_std = to_column((x_grid_raw - data["mean"]) / data["std"])

    # 三种损失用固定的颜色，后面几张子图保持一致，读者一眼能对上。
    loss_colors = {"mse": "#1f77b4", "mae": "#ff7f0e", "huber": "#2ca02c"}

    # ---- 子图 (a)：三种损失的训练曲线 ----
    # 注意：MSE / MAE / Huber 的数值不能直接比较大小（量纲不同），
    # 所以这张图只看"各自有没有收敛、收敛得快不快"，不看谁高谁低。
    for loss_name in ["mse", "mae", "huber"]:
        hist = results[loss_name]["history"]
        axes[0].plot(hist["epoch"], hist["train_loss"],
                     color=loss_colors[loss_name], linewidth=2.0,
                     label="%s" % loss_name.upper())
    axes[0].set_yscale("log")        # 对数轴，方便同时看清前期的快速下降和后期的稳定
    axes[0].set_xlabel("迭代轮数 epoch")
    axes[0].set_ylabel("训练集 Loss（各自的尺度，对数刻度）")
    axes[0].set_title("(a) 三种损失的收敛曲线（mini-batch=16, lr=0.05）")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, which="both")

    # ---- 子图 (b)：被污染的数据 + 三条拟合直线 ----
    # 用一个布尔数组标出哪些样本是异常点。
    is_outlier = np.zeros(len(y_train), dtype=bool)     # 全 False
    for idx in outlier_indices:
        is_outlier[idx] = True                          # 把异常点标成 True

    # 正常数据点（浅蓝色小圆点）。~is_outlier 是"取反"，也就是"不是异常点的那些"。
    axes[1].scatter(data["x_train_raw"].reshape(-1)[~is_outlier],
                    y_train.reshape(-1)[~is_outlier],
                    s=24, color=C_TRAIN, alpha=0.5, label="正常数据点")

    # 异常点（红色大叉）。
    axes[1].scatter(data["x_train_raw"].reshape(-1)[is_outlier],
                    y_train.reshape(-1)[is_outlier],
                    s=110, marker="x", color=C_TEST, linewidths=2.4,
                    label="人工注入的异常点", zorder=5)

    # 三条直线。
    for loss_name in ["mse", "mae", "huber"]:
        w_std = results[loss_name]["w_std"]
        b_std = results[loss_name]["b_std"]
        y_grid = (x_grid_std * w_std + b_std).reshape(-1)     # 在这条网格上做预测
        axes[1].plot(x_grid_raw, y_grid, linewidth=2.3,
                     color=loss_colors[loss_name], label="%s" % loss_name.upper())

    # 顺便画一下"没有异常值时"的正确直线（黑色虚线），一眼就能看出谁偏得多。
    axes[1].plot(x_grid_raw, w_clean_raw * x_grid_raw + b_clean_raw,
                 color="black", linestyle=":", linewidth=1.8,
                 label="无异常值的正确直线", zorder=4)

    axes[1].set_xlabel("学习时长 hours")
    axes[1].set_ylabel("成绩 score（含异常值）")
    axes[1].set_title("(b) 异常值存在时，三种损失学到的直线")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # ---- 子图 (c)：斜率被带偏了多少（柱状图 + 错误率标注） ----
    names = ["mse", "mae", "huber"]                       # 三种损失的名字
    slopes = [results[n]["w_raw"] for n in names]         # 对应的斜率

    # 画柱状图。
    bars = axes[2].bar(names, slopes,
                       color=[loss_colors[n] for n in names], alpha=0.85, width=0.6)

    # 在每根柱子上标出斜率数值和偏离百分比。
    for bar, loss_name in zip(bars, names):
        value = results[loss_name]["w_raw"]
        err = results[loss_name]["slope_error_pct"]
        axes[2].text(bar.get_x() + bar.get_width() / 2.0, value + 0.06,
                     "%.3f\n(偏 %.1f%%)" % (value, err),
                     ha="center", fontsize=9)

    # 画"正确斜率"的参考线。
    axes[2].axhline(w_clean_raw, color="black", linestyle="--", linewidth=1.8,
                    label="无异常值时的正确斜率 %.3f" % w_clean_raw)

    # 把纵轴范围留够，免得标注文字被切掉。
    axes[2].set_ylim(0, max(slopes) * 1.25)

    axes[2].set_ylabel("拟合直线的斜率 w (score / hour)")
    axes[2].set_title("(c) 谁被异常值带偏得最少？")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, axis="y")     # 柱子图只在 y 方向画网格比较好看

    # 保存。
    save_figure(fig, "07_loss_functions.png")

    # 返回结果。
    return results


def experiment_momentum_decay(data):
    """
    实验 F：动量（Momentum）和学习率衰减（LR Decay）到底有没有用？

    我们做一个"消融实验（ablation study）"：
    从同一个基线出发，每次只加一个技巧，看它单独带来了多少提升。
    这是机器学习论文里最常见的实验设计方式。

    ⚠️ 这个实验里我们踩了第二个坑（详见 note.md 与报告）：
        一开始我用 lr=0.6 做对比，发现"加了动量反而更慢"，一度以为动量写错了。
        反复检查代码没问题后，我才想清楚背后的道理：
            在我们采用的 v = βv + (1-β)g 这种写法里，动量的"有效步长"
            约等于 lr/(1-β)。当 β=0.9 时相当于把学习率放大了 10 倍！
            所以在 lr=0.6 这种"本来就偏大"的情况下，加动量等于把步长推到 6，
            必然剧烈震荡；而基线虽然慢，却没有震荡，反而先到。
        这说明动量的价值不在"随便加就更快"，而在：
            ① 在学习率接近稳定边界时"救场"（大学习率 + 动量反而能收敛）；
            ② 配合学习率衰减一起用，前期猛冲、后期精细收敛。
        所以下面的实验设计成三组，把这两点都展示出来。
    """

    print("\n[实验 F] 动量与学习率衰减的消融实验")

    # 四个配置：(显示名字, 动量系数, 学习率衰减)
    configs = [
        ("基线 (纯 BGD)", 0.0, 1.00),          # 什么都不加
        ("+ 动量 0.9", 0.9, 1.00),             # 只加动量
        ("+ 学习率衰减 0.99", 0.0, 0.99),      # 只加衰减
        ("+ 动量 + 衰减", 0.9, 0.99),          # 两个都加
    ]

    # 三个场景，对应三种不同的学习率（epochs 用同一个值，保证对比公平）。
    lr_gentle = 0.1     # 场景一：温和学习率 —— 基线本来就走得动，动量在这里帮不上忙
    lr_edge = 1.0       # 场景二：刚好压在稳定边界上 —— 基线直接爆炸，动量来救场
    lr_decay = 0.5      # 场景三：配合学习率衰减 —— 前期大步、后期小步，最稳最快
    epochs = 150        # 三种场景都用 150 轮

    # 四个配置的固定颜色（提前定义，后面算指标和画图都要用）。
    cfg_colors = {
        "基线 (纯 BGD)": "#1f77b4",
        "+ 动量 0.9": "#ff7f0e",
        "+ 学习率衰减 0.99": "#2ca02c",
        "+ 动量 + 衰减": "#d62728",
    }

    # 用嵌套字典保存结果：results[场景名][配置名] = history
    results = {}

    # 先把三个场景的名字和对应学习率列出来，方便后面循环。
    scenarios = [
        ("温和 lr=0.1", lr_gentle),
        ("临界 lr=1.0", lr_edge),
        ("衰减 lr=0.5", lr_decay),
    ]

    for scene_name, lr in scenarios:
        results[scene_name] = {}

        for name, mom, decay in configs:
            print("  --- %s, %s ---" % (scene_name, name))

            # 新模型（每个配置都从头开始，避免互相污染）。
            model = LinearRegression()

            # 训练。用批量梯度下降（全量），排除批量噪声的干扰，
            # 这样三条曲线的差别就只能来自"动量"和"衰减"。
            history = gradient_descent(
                model,
                data["x_train_std"], data["y_train"],
                data["x_test_std"], data["y_test"],
                learning_rate=lr,        # ← 场景变量
                epochs=epochs,
                batch_size=None,         # 全量，排除批量噪声的干扰
                loss_name="mse",
                momentum=mom,            # ← 自变量之一
                lr_decay=decay,          # ← 自变量之二
                seed=42,
                record_test=False,
            )

            # 记录。
            results[scene_name][name] = history
            print("      最终 train MSE = %.4f" % history["train_loss"][-1])

    # ---------- 计算"多快能到达最优附近"，这是评价优化器好坏的关键指标 ----------
    # 为什么不能只看"最后一轮的 loss"？
    #   越接近最优点，loss 每一轮的波动可能比不同配置之间的差别还大，
    #   看最后一轮很容易被随机波动误导。
    #   所以更科学的指标是"第几轮进入了最优 loss 的 1.02 倍以内"。
    w_cf, b_cf = fit_closed_form(data["x_train_std"], data["y_train"])
    model_cf = LinearRegression()
    model_cf.set_params(w_cf, b_cf)
    loss_best = float(mse_loss(model_cf.predict(data["x_train_std"]), data["y_train"]))

    # 目标线：最优 loss 的 1.02 倍。
    target = loss_best * 1.02

    # reach_info[场景][配置] = 第几轮达标（未达标则为 None）
    reach_info = {}

    print("  目标线 = 最优 loss × 1.02 = %.2f，各配置第几轮能达标？" % target)

    for scene_name, _ in scenarios:
        reach_info[scene_name] = {}
        for name in cfg_colors:
            # 取出这条曲线的训练 loss。
            curve = results[scene_name][name]["train_loss"]

            # 找第一个 <= 目标线的位置。
            hit = np.where(curve <= target)[0]

            # 记录了轮数（下标 + 1）；没找到就是 None。
            reach_info[scene_name][name] = (int(hit[0]) + 1) if hit.size > 0 else None

        # 打印这一场景的对比表。
        print("      【%s】" % scene_name)
        for name in cfg_colors:
            ep = reach_info[scene_name][name]
            print("          %-20s → %s"
                  % (name, "始终未达标（不收敛）" if ep is None else "第 %d 轮达标" % ep))

    # ---------------- 画图 8：三个场景 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # ---- 子图 (a)：温和学习率，看"基线是不是本来就走得动" ----
    for name in cfg_colors:
        hist = results["温和 lr=0.1"][name]
        # 只画前 60 轮，把差别放大。
        axes[0].plot(hist["epoch"][:60], hist["train_loss"][:60],
                     linewidth=2.2, color=cfg_colors[name], label=name)

    axes[0].axhline(target, color="black", linestyle="--", linewidth=1.2,
                    label="目标线 = 最优×1.02")
    axes[0].set_xlabel("迭代轮数 epoch（只画前 60 轮）")
    axes[0].set_ylabel("训练集 Loss (MSE)")
    axes[0].set_title("(a) 温和 lr=%g：基线本来就走得动，动量帮助有限" % lr_gentle)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # ---- 子图 (b)：临界学习率，动量救场 ----
    for name in cfg_colors:
        hist = results["临界 lr=1.0"][name]
        # 这里纵轴用对数，才能同时看清"爆炸到几千"和"收敛到 25"。
        y_plot = clip_curve(hist["train_loss"][:40], 1e5)
        axes[1].plot(hist["epoch"][:40], y_plot,
                     linewidth=2.2, color=cfg_colors[name], label=name)

    axes[1].axhline(target, color="black", linestyle="--", linewidth=1.2,
                    label="目标线 = 最优×1.02")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("迭代轮数 epoch（只画前 40 轮）")
    axes[1].set_ylabel("训练集 Loss (MSE, 对数刻度)")
    axes[1].set_title("(b) 临界 lr=%g：基线直接爆炸，动量出手救场" % lr_edge)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    # ---- 子图 (c)：达标轮数柱状图（跨三个场景横向对比） ----
    names_bar = list(cfg_colors.keys())          # 四个配置名
    n_cfg = len(names_bar)                       # 4
    width = 0.26                                 # 柱宽

    # 配置的横向位置。
    x_pos = np.arange(n_cfg)

    # 三个场景各占一个偏移量：-width, 0, +width。
    scene_offsets = [-width, 0.0, width]

    # 三个场景用同一套颜色，但用不同的透明度区分，避免颜色太多看花眼。
    scene_alphas = [1.0, 0.7, 0.4]

    for si, (scene_name, _) in enumerate(scenarios):
        # 这个场景下，四个配置的达标轮数；未达标的用总轮数代替（柱子会很高）。
        vals = []
        for name in names_bar:
            ep = reach_info[scene_name][name]
            vals.append(ep if ep is not None else epochs)

        # 画柱子。
        bars_s = axes[2].bar(x_pos + scene_offsets[si], vals, width,
                             color=[cfg_colors[n] for n in names_bar],
                             alpha=scene_alphas[si],
                             edgecolor="black", linewidth=0.5,
                             label=scene_name)

        # 在每个柱顶标出轮数（未达标就标"×"）。
        for xi, name in zip(x_pos + scene_offsets[si], names_bar):
            ep = reach_info[scene_name][name]
            text = "×" if ep is None else "%d" % ep
            axes[2].text(xi, (ep if ep is not None else epochs) + 2.0,
                         text, ha="center", fontsize=7.5)

    # 横轴标签换成简短的配置名。
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(["基线", "+\n动量", "+\n衰减", "动量\n+衰减"], fontsize=9)

    axes[2].set_ylabel("达标所需轮数（越少越快）")
    axes[2].set_title("(c) 达标轮数对比（× = 150 轮内始终不收敛）")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, axis="y")

    # 保存。
    save_figure(fig, "08_momentum_decay.png")

    # 返回结果。
    return results


def experiment_test_fit(data, final_config):
    """
    实验 G（最终交付）：用调好的配置训练最终模型，并在【测试集】上评估。

    这是整个项目最后一步，也是最重要的一步：
    我们只看这一次的测试集结果，把它当作"模型的真实水平"。
    （如果在调参过程中反复看测试集，测试集就被"用脏"了，
      严格来说应该另留一份验证集。本项目数据量小，我们采用
      "用训练集损失曲线选超参数 + 最后在测试集上评估一次"的务实做法，
      并在报告里说明这个局限。）
    """

    print("\n[实验 G] 最终模型：在测试集上评估拟合效果")

    # 打印配置，方便对照。
    print("  最终配置:", final_config)

    # 新模型。
    model = LinearRegression()

    # 用最终配置训练。
    history = gradient_descent(
        model,
        data["x_train_std"], data["y_train"],
        data["x_test_std"], data["y_test"],
        learning_rate=final_config["learning_rate"],
        epochs=final_config["epochs"],
        batch_size=final_config["batch_size"],
        loss_name=final_config["loss_name"],
        momentum=final_config["momentum"],
        lr_decay=final_config["lr_decay"],
        seed=final_config["seed"],
        verbose=True,          # 打印训练进度，让人看到训练确实在进行
    )

    # 取参数并换算回原始空间。
    w_std, b_std = model.get_params()
    w_raw, b_raw = restore_params(w_std, b_std, data["mean"], data["std"])

    # 在训练集和测试集上分别预测。
    pred_train = model.predict(data["x_train_std"])
    pred_test = model.predict(data["x_test_std"])

    # 计算指标。
    metrics_train = evaluate_all_metrics(data["y_train"], pred_train)
    metrics_test = evaluate_all_metrics(data["y_test"], pred_test)

    # 打印最终模型。
    print("  最终模型: score = %.4f * hours + %.4f" % (w_raw, b_raw))
    print("  训练集: MSE=%.4f RMSE=%.4f MAE=%.4f R2=%.4f"
          % (metrics_train["mse"], metrics_train["rmse"],
             metrics_train["mae"], metrics_train["r2"]))
    print("  测试集: MSE=%.4f RMSE=%.4f MAE=%.4f R2=%.4f"
          % (metrics_test["mse"], metrics_test["rmse"],
             metrics_test["mae"], metrics_test["r2"]))

    # ---------------- 画图 9：测试集拟合效果 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # 左图：测试集散点 + 拟合直线。
    axes[0].scatter(data["x_test_raw"].reshape(-1), data["y_test"].reshape(-1),
                    s=55, color=C_TEST, alpha=0.8, label="测试集真实值", zorder=3)

    # 画拟合直线。
    x_grid_raw = np.linspace(data["x_train_raw"].min(), data["x_test_raw"].max(), 100)
    axes[0].plot(x_grid_raw, w_raw * x_grid_raw + b_raw,
                 color=C_LINE, linewidth=2.5,
                 label="模型预测: score=%.2f*hours%+.2f" % (w_raw, b_raw), zorder=2)

    axes[0].set_xlabel("学习时长 hours")
    axes[0].set_ylabel("成绩 score")
    axes[0].set_title("(a) 测试集拟合效果 (R² = %.4f)" % metrics_test["r2"])
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # 右图：预测值 vs 真实值的散点图。
    # 理想情况下这些点应该排成一条 45° 的对角线（预测=真实）。
    axes[1].scatter(data["y_test"].reshape(-1), pred_test.reshape(-1),
                    s=55, color=C_TRAIN, alpha=0.8, label="测试集样本", zorder=3)

    # 画 y=x 的参考对角线。
    # 取所有 y 的最小值和最大值作为线的两端。
    lo = float(min(data["y_test"].min(), pred_test.min())) - 3.0
    hi = float(max(data["y_test"].max(), pred_test.max())) + 3.0
    axes[1].plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.4,
                 label="理想情况 y = ŷ", zorder=2)

    axes[1].set_xlabel("真实值 y (score)")
    axes[1].set_ylabel("预测值 ŷ (score)")
    axes[1].set_title("(b) 预测值 vs 真实值 (点越贴近对角线越好)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # 保存。
    save_figure(fig, "09_test_fit.png")

    # ---------------- 画图 10：残差分析 ----------------
    # 残差 = 真实值 - 预测值。看残差能发现"模型是不是系统性地高估/低估"。
    residuals = data["y_test"] - pred_test

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    # 左图：残差 vs 预测值（或特征）。理想情况是随机散布在 0 附近，没有明显形状。
    axes[0].scatter(data["x_test_raw"].reshape(-1), residuals.reshape(-1),
                    s=50, color="#9467bd", alpha=0.8, zorder=3)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.4, label="残差 = 0")
    axes[0].set_xlabel("学习时长 hours")
    axes[0].set_ylabel("残差 = 真实值 - 预测值")
    axes[0].set_title("(a) 残差分布：若无明显规律说明线性假设基本成立")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # 右图：残差直方图，看误差是不是大致对称、中心在 0。
    # bins=12 表示把数据分成 12 个区间来统计。
    axes[1].hist(residuals.reshape(-1), bins=12, color="#8c564b",
                 alpha=0.8, edgecolor="white")

    # 画一条 0 参考线。
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.4)

    axes[1].set_xlabel("残差 (score)")
    axes[1].set_ylabel("样本个数")
    axes[1].set_title("(b) 残差直方图：均值 %.2f，标准差 %.2f"
                      % (float(residuals.mean()), float(residuals.std())))
    axes[1].grid(alpha=0.3, axis="y")

    # 保存。
    save_figure(fig, "10_residuals.png")

    # 把预测结果整理成表格行，保存成 CSV，方便人工核对。
    rows = []                                  # 每一行：样本序号, hours, 真实值, 预测值, 残差
    for i in range(data["n_test"]):
        # 逐个样本取出数值，都转成 float。
        rows.append([
            i + 1,                                              # 序号（从 1 开始，给人看）
            round(float(data["x_test_raw"][i, 0]), 3),           # 原始 hours
            round(float(data["y_test"][i, 0]), 3),               # 真实 score
            round(float(pred_test[i, 0]), 3),                    # 预测 score
            round(float(residuals[i, 0]), 3),                    # 残差
        ])

    # 保存成 CSV。
    write_csv("test_predictions.csv",
              ["index", "hours", "true_score", "pred_score", "residual"],
              rows)

    # 返回最终结果，供保存 JSON 使用。
    return {
        "history": history,                 # 训练历史
        "w_raw": float(w_raw),              # 原始空间斜率
        "b_raw": float(b_raw),              # 原始空间截距
        "w_std": float(w_std),              # 标准化空间斜率
        "b_std": float(b_std),              # 标准化空间截距
        "metrics_train": metrics_train,     # 训练集指标
        "metrics_test": metrics_test,       # 测试集指标
        "config": final_config,             # 使用的配置
    }


# ---------------------------- 7. 主函数：把所有实验串起来 ----------------------------

def main():
    """
    主函数：按顺序执行所有实验，并保存结果。
    """

    # ---------- 7.1 解析命令行参数 ----------
    # argparse 是 Python 标准库里用来处理命令行参数的模块。
    parser = argparse.ArgumentParser(
        description="从零实现线性回归 + 梯度下降（NumPy）")

    # 每个 add_argument 定义一个新的可选参数。
    parser.add_argument("--epochs", type=int, default=200,
                        help="学习率对比实验的迭代轮数（默认 200）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，保证结果可复现（默认 42）")
    parser.add_argument("--no-figures", action="store_true",
                        help="不把图片写进 results/（只跑训练和统计，速度更快）")
    parser.add_argument("--log", type=str, default="",
                        help="把完整输出同时写进这个日志文件（默认不写日志）")

    # parse_args() 真正去读命令行。例如 --epochs 300 会被解析成 args.epochs = 300。
    args = parser.parse_args()

    # 如果加了 --no-figures，就把"保存图片"的全局开关关掉。
    # 注意：这里要声明 global，否则赋值只会创建一个局部变量，外面的开关不会变。
    global SAVE_FIGURES
    if args.no_figures:
        SAVE_FIGURES = False

    # 顺便把 matplotlib 的交互式功能关掉（其实 Agg 后端已经关了，
    # 这里只是让意图更明确）。
    if args.no_figures:
        plt.ioff()

    # ---------- 7.2 准备环境 ----------
    # 建结果目录（已存在也不会报错）。日志文件也要放进这个目录，
    # 所以这一步必须在"开启日志"之前完成。
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 如果命令行里给了 --log，就开启"屏幕 + 文件"双份输出。
    original_stdout = None      # 一开始假设没有开启日志
    log_file = None

    # 判断用户有没有传 --log。
    if args.log:
        # 如果传的是相对路径，就把它放到 results/ 目录下面，避免文件乱丢。
        log_path = args.log
        if not os.path.isabs(log_path):
            log_path = os.path.join(RESULTS_DIR, log_path)

        # 开启日志，并记住需要还原的东西。
        original_stdout, log_file = start_logging(log_path)

    # 下面的所有内容都放在 try 里，配合最后的 finally，
    # 保证无论训练成功还是中途报错，日志文件都会被正确关闭。
    try:
        # 打印欢迎信息，让人知道程序开始跑了。
        print("=" * 72)
        print("从零实现线性回归 + 梯度下降（NumPy 版）")
        print("训练集: data/train.csv   测试集: data/test.csv")
        print("=" * 72)

        # 设置中文字体。
        setup_matplotlib_for_chinese()

        # 固定随机种子，并拿到一个随机数生成器。
        rng = set_seed(args.seed)

        # 打印一下 Python / NumPy 版本，方便复现问题时排查环境。
        print("Python 版本:", sys.version.split()[0])
        print("NumPy 版本:", np.__version__)

        # ---------- 7.3 加载数据 ----------
        print("\n[数据] 正在读取并预处理数据...")

        # prepare_data 会读 CSV、标准化、整理形状，一条龙完成。
        data = prepare_data()

        # 打印数据概况。
        print("  训练集: %d 个样本   测试集: %d 个样本"
              % (data["n_train"], data["n_test"]))
        print("  特征 hours 的范围: [%.2f, %.2f]"
              % (data["x_train_raw"].min(), data["x_train_raw"].max()))
        print("  标准化参数: mean = %.4f, std = %.4f" % (data["mean"], data["std"]))

        # 准备一个字典，收集所有实验的关键数字，最后存成 results.json。
        summary = {}

        # ---------- 7.4 逐个执行实验 ----------
        # 实验 0：闭式解基线。
        baseline = experiment_baseline(data, rng)
        summary["baseline"] = baseline

        # 实验 A：loss 曲线 + 拟合过程。
        # 注意：这个实验本身包含"训练"，所以无论是否保存图片都要跑；
        #       --no-figures 只是让 save_figure 不落盘而已。
        history_a = experiment_loss_curve(data)
        summary["loss_curve_final_train_mse"] = float(history_a["train_loss"][-1])

        # 实验 B：学习率对比。
        # 精心挑选的学习率列表（MSE 的梯度是 2*(误差)*x，数值比较大，
        # 所以在"标准化 + 全量"这种最理想的情况下，可用的学习率上限大约在 1 附近）：
        #   0.001 / 0.01 → 太小，200 轮还没走到最低点
        #   0.05 / 0.1   → 下降又快又稳
        #   0.3 / 0.5    → 仍然收敛，但开始出现"一步跨过头再弹回来"的震荡
        #   0.9 / 1.2    → 临界与发散
        lr_list = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 1.2]
        histories_std, final_losses = experiment_learning_rate(data, lr_list)

        # 把每个学习率最终的 loss 存进 summary（nan 单独处理，因为 JSON 不支持 NaN）。
        lr_final = {}
        for lr, fl in zip(lr_list, final_losses):
            # 如果是 nan 或 inf，就存成字符串 "diverged"，其余存数字。
            if not np.isfinite(fl):
                lr_final["%g" % lr] = "diverged(nan)"
            else:
                lr_final["%g" % lr] = float(fl)
        summary["learning_rate_final_loss"] = lr_final

        # 找出表现最好的学习率（忽略发散的那些）。
        best_lr = None                 # 目前最好的学习率
        best_loss = float("inf")       # 目前最好的 loss（先设成无穷大）
        for lr, fl in zip(lr_list, final_losses):
            # 只有有限值才参与比较。
            if np.isfinite(fl) and fl < best_loss:
                best_loss = fl
                best_lr = lr
        print("\n  结论：表现最好的学习率是 lr = %g （最终 train MSE = %.4f）"
              % (best_lr, best_loss))
        summary["best_learning_rate"] = best_lr
        summary["best_learning_rate_final_mse"] = float(best_loss)

        # 实验 C：标准化对比。
        histories_raw = experiment_standardization(data, histories_std, lr_list)

        # 把"原始尺度 vs 标准化"在每个学习率下的最终 loss 记录下来。
        # 报告里那张"标准化拓宽了可用学习率范围"的图，数据就来自这里。
        std_compare = {}
        for lr in lr_list:
            std_compare["%g" % lr] = {
                "raw_scale_final_mse": json_safe(np.float64(
                    histories_raw[lr]["train_loss"][-1])),
                "standardized_final_mse": json_safe(np.float64(
                    histories_std[lr]["train_loss"][-1])),
            }
        summary["standardization_compare"] = std_compare

        # 记录一条结论性的文字说明，报告里会用到。
        summary["standardization_note"] = (
            "原始尺度在 lr>=0.05 时很快发散；标准化后同样的 lr 可以稳定收敛"
        )

        # 实验 D：批量大小对比。
        # 这里安排了 4 组，故意做成"两两对照"，结论才站得住脚：
        #   前两组用【同一个学习率】，看的是"每一步的梯度噪声"有多大 → mini-batch 平滑、SGD 抖
        #   后两组让【总更新次数尽量接近】，看的是"最终收敛到哪" → 各调各的学习率才公平
        batch_configs = [
            ("BGD (batch=100, lr=0.05)", None, 0.05),          # 全量：每步都看 100 个样本
            ("Mini-batch (batch=16, lr=0.05)", 16, 0.05),      # 小批量：每步看 16 个
            ("SGD (batch=1, lr=0.05)", 1, 0.05),               # 随机：每步只看 1 个（噪声最大）
            ("SGD (batch=1, lr=0.01)", 1, 0.01),               # 单独调小学习率的 SGD
        ]
        batch_results = experiment_batch_size(data, batch_configs)

        # 记录每种批量大小最终的 loss 和更新次数。
        batch_summary = {}
        for name, _, _ in batch_configs:
            batch_summary[name] = {
                "final_train_mse": float(batch_results[name]["train_loss"][-1]),
                "num_updates": int(batch_results[name]["step"][-1]),
            }
        summary["batch_size"] = batch_summary

        # 实验 E：损失函数抗异常值能力。
        # 故意挑 4 个样本（例如低 hours 但高分），把成绩硬加 45 分，制造异常。
        outlier_indices = [3, 12, 25, 40]       # 训练集里的第 4、13、26、41 个样本
        outlier_shift = 45.0                    # 加 45 分
        loss_results = experiment_loss_functions(data, outlier_indices, outlier_shift)

        # 记录三种损失学到的斜率和"被带偏的百分比"，报告里要用。
        loss_slopes = {}
        for loss_name in ["mse", "mae", "huber"]:
            loss_slopes[loss_name] = {
                "w": loss_results[loss_name]["w_raw"],                     # 原始空间斜率
                "b": loss_results[loss_name]["b_raw"],                     # 原始空间截距
                "slope_error_pct": loss_results[loss_name]["slope_error_pct"],  # 偏离正确斜率的百分比
            }
        summary["loss_function_slopes_with_outliers"] = loss_slopes
        summary["outlier_setup"] = {
            "indices": outlier_indices,
            "shift": outlier_shift,
        }

        # 实验 F：动量 + 学习率衰减。
        momentum_results = experiment_momentum_decay(data)

        # 记录三个场景、四种配置的最终 loss 和"达标轮数"。
        # 需要先算出"目标损失"（最优 loss 的 1.02 倍），才能统计达标轮数。
        w_cf3, b_cf3 = fit_closed_form(data["x_train_std"], data["y_train"])
        model_cf3 = LinearRegression()
        model_cf3.set_params(w_cf3, b_cf3)
        loss_best3 = float(mse_loss(model_cf3.predict(data["x_train_std"]), data["y_train"]))
        target3 = loss_best3 * 1.02

        summary["momentum_decay"] = {
            "target_loss": float(target3),                                      # 目标线
            "table": momentum_final_and_reach(momentum_results, target3),       # 对比表
        }

        # 实验 G：最终模型 + 测试集评估。
        # 这个配置是这样选出来的（每一步都有前面的实验作为依据）：
        #   - 损失函数用 MSE：任务要求，而且这份数据里的异常值不多，
        #     MSE 能让"错得多的样本"获得更多关注，拟合更准；
        #   - 学习率 0.05：实验 B 显示 0.05~0.9 都能收敛，但 0.05 更稳，
        #     而且接下来要加动量（会把有效步长放大），所以底子要稳一点；
        #   - 动量 0.9：实验 F 证明它能在大学习率下"救场"、并加速收敛；
        #   - 学习率衰减 0.998：前期大步快跑，后期小步精调，最后几步更接近最优点；
        #   - 小批量 16：实验 D 显示它在"梯度平稳"和"更新次数多"之间平衡得最好。
        final_config = {
            "learning_rate": 0.05,      # 学习率
            "epochs": 300,              # 迭代轮数
            "batch_size": 16,           # 小批量大小
            "loss_name": "mse",         # 损失函数
            "momentum": 0.9,            # 动量系数
            "lr_decay": 0.998,          # 学习率衰减系数（每轮乘一次）
            "seed": args.seed,          # 随机种子
        }
        final_result = experiment_test_fit(data, final_config)

        # 把最终结果记进 summary。
        summary["final_model"] = {
            "config": final_config,
            "w_raw": final_result["w_raw"],
            "b_raw": final_result["b_raw"],
            "w_std": final_result["w_std"],
            "b_std": final_result["b_std"],
            "metrics_train": final_result["metrics_train"],
            "metrics_test": final_result["metrics_test"],
        }

        # ---------- 7.5 保存汇总结果 ----------
        # 把 summary 存成 JSON 文件。indent=2 让文件有缩进、方便人看。
        json_path = os.path.join(RESULTS_DIR, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            # ensure_ascii=False 让中文正常显示，而不是变成 \uXXXX 转义。
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("\n[汇总] 所有关键指标已保存:", os.path.relpath(json_path, _PROJECT_ROOT))

        # ---------- 7.6 保存 loss 曲线的原始数据 ----------
        # 把最终模型的训练曲线也存成 CSV，report 里如果需要精确数字可以直接查。
        hist = final_result["history"]
        curve_rows = []                         # 每一行：epoch, train_mse, test_mse, lr
        for i in range(len(hist["epoch"])):
            curve_rows.append([
                int(hist["epoch"][i]),                    # 第几轮
                round(float(hist["train_loss"][i]), 6),   # 训练集 MSE
                round(float(hist["test_loss"][i]), 6),    # 测试集 MSE
                round(float(hist["lr_used"][i]), 6),      # 这一轮的学习率
            ])
        write_csv("final_training_curve.csv",
                  ["epoch", "train_mse", "test_mse", "learning_rate"], curve_rows)

        # ---------- 7.7 打印最终总结 ----------
        print("\n" + "=" * 72)
        print("全部实验完成！最终模型：")
        print("    score = %.4f * hours + %.4f"
              % (final_result["w_raw"], final_result["b_raw"]))
        print("    测试集 R² = %.4f, RMSE = %.4f, MAE = %.4f"
              % (final_result["metrics_test"]["r2"],
                 final_result["metrics_test"]["rmse"],
                 final_result["metrics_test"]["mae"]))
        print("所有结果已保存到:", os.path.relpath(RESULTS_DIR, _PROJECT_ROOT))
        print("=" * 72)

    finally:
        # finally 里的代码"无论上面成功还是报错都会执行"，
        # 这是保证"日志文件一定会被正确关闭"的标准写法。
        if log_file is not None:
            stop_logging(original_stdout, log_file)


# =============================================================================
# 程序入口
# =============================================================================
# 这一行保证：直接运行 python train.py 时会执行 main()，
# 而当 train.py 被别的脚本 import 时不会自动开跑（方便做单元测试）。
if __name__ == "__main__":
    main()
