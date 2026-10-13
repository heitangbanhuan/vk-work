# -*- coding: utf-8 -*-
# =============================================================================
# model.py  ——  线性回归模型（从零实现，只用 NumPy）
# -----------------------------------------------------------------------------
# 这个文件负责"模型"和"损失函数"这两件事，一句话概括就是：
#
#     给定数据 x 和参数 (w, b)，算出预测值 y_hat；
#     再把 y_hat 和真实值 y 一比，算出"错得有多离谱"（loss）；
#     最后算出"参数应该往哪个方向调"（梯度 gradient）。
#
# 注意：我们【没有】使用 sklearn 的 LinearRegression，
# 而是自己写公式、自己推梯度。这正是本任务的核心要求。
#
# 本文件出现的数学符号约定（和报告里的符号保持一致）：
#     n : 一批样本的个数
#     x : 输入特征，形状 (n, 1)
#     y : 真实目标值，形状 (n, 1)
#     w : 权重（斜率），形状 (1, 1)，即一个数；写成矩阵是为了方便做矩阵乘法
#     b : 偏置（截距），形状 (1, 1)，即一个数
#     y_hat : 模型预测值，形状 (n, 1)
# =============================================================================

# numpy：唯一用到的第三方库。本项目的核心要求就是"不调用高层封装"。
import numpy as np


# =============================================================================
# 第一部分：前向传播（Forward）—— 用当前参数算出预测值
# =============================================================================
def forward(x, w, b):
    """
    线性回归的模型形式：

        y_hat = x @ w + b          （@ 在 NumPy 里表示矩阵乘法）

    如果只有一个特征，按标量写出来就是大家最熟悉的样子：

        y_hat = w * x + b

    参数:
        x : 特征矩阵，形状 (n, 1)
        w : 权重矩阵，形状 (1, 1)
        b : 偏置矩阵，形状 (1, 1)

    返回:
        y_hat : 预测值，形状 (n, 1)
    """

    # x @ w 是矩阵乘法：(n, 1) 乘 (1, 1) 得到 (n, 1)。
    # 因为 b 的形状是 (1, 1)，NumPy 会自动把它"广播（broadcast）"到每一行，
    # 相当于给每个样本都加上了同一个截距，不用我们写循环。
    y_hat = x @ w + b

    # 返回预测值。
    return y_hat


# =============================================================================
# 第二部分：三种损失函数（Loss）
# =============================================================================
# 损失函数回答的问题是："模型现在错得有多离谱？"
# 训练的目标就是不断调整 w、b，让这个数字越小越好。
# 我们实现了三种，方便在实验里对比它们对"异常值"的敏感程度。
# =============================================================================

def mse_loss(y_hat, y):
    """
    均方误差 MSE（Mean Squared Error）—— 本任务的主损失函数。

        MSE = (1/n) * Σ (y_hat_i - y_i)^2

    含义：把每个样本的误差平方后取平均。
    为什么用平方？两个原因：
        1) 正负误差不会互相抵消；
        2) 大误差被"放大"，模型会优先去修正错得最狠的样本。
    MSE 对 y_hat 是凸函数，所以梯度下降一定能走到全局最优（不会卡在局部最优）。

    参数:
        y_hat : 预测值，形状 (n, 1)
        y     : 真实值，形状 (n, 1)

    返回:
        loss : 一个标量（float），越小越好
    """

    # y_hat - y 是每个样本的误差，形状仍然是 (n, 1)。
    diff = y_hat - y

    # diff ** 2 把每个元素平方；.mean() 再对所有元素求平均。
    # 两步合起来就是 MSE。
    loss = (diff ** 2).mean()

    # 返回标量损失。
    return loss


def mse_grad(y_hat, y, x):
    """
    MSE 对参数 w 和 b 的梯度（这是本任务最核心的一步推导）。

    推导过程（用链式法则，只有一个特征，所以可以写成标量形式）：

        定义单个样本的损失：   L_i = (y_hat_i - y_i)^2
        而 y_hat_i = w * x_i + b

        对 w 求偏导：
            ∂L_i/∂w = 2 * (y_hat_i - y_i) * ∂y_hat_i/∂w
                    = 2 * (y_hat_i - y_i) * x_i

        对 b 求偏导：
            ∂L_i/∂b = 2 * (y_hat_i - y_i) * ∂y_hat_i/∂b
                    = 2 * (y_hat_i - y_i) * 1

        对整个数据集（n 个样本）取平均，并"约掉" 2/n 里的常数：
        实践中习惯把常数 2 一起吸收进学习率里（写不写 2 只是学习率差一倍），
        为了公式干净，我们采用下面这种最常见的写法：

            ∂L/∂w = (2/n) * Σ (y_hat_i - y_i) * x_i
            ∂L/∂b = (2/n) * Σ (y_hat_i - y_i)

    用矩阵形式写，就是本函数返回的两行：
        dw = (2/n) * x.T @ diff
        db = (2/n) * sum(diff)

    参数:
        y_hat : 预测值，形状 (n, 1)
        y     : 真实值，形状 (n, 1)
        x     : 这一批的特征，形状 (n, 1)

    返回:
        dw : 损失对 w 的梯度，形状 (1, 1)
        db : 损失对 b 的梯度，形状 (1, 1)
    """

    # n 是这一批样本的个数。注意是"这一批"，不是整个数据集——
    # 这样批量梯度下降和小批量梯度下降才能共用同一个函数。
    n = x.shape[0]

    # 误差 diff = y_hat - y，形状 (n, 1)。
    diff = y_hat - y

    # x.T 是 x 的转置，把 (n,1) 变成 (1,n)。
    # (1,n) @ (n,1) = (1,1)，正好把 n 个样本的误差加权求和，
    # 权重就是各自的 x_i。这就是 Σ (y_hat_i - y_i) * x_i。
    # 乘 2/n 完成"求平均 + 系数 2"。
    dw = (2.0 / n) * (x.T @ diff)

    # 对 b 的偏导不需要乘 x_i，所以直接把误差全部加起来再乘 2/n。
    # .sum() 会把 (n,1) 的所有元素加起来得到一个标量，
    # reshape(1, 1) 是为了让它和 b 的形状 (1,1) 保持一致，方便相减。
    db = (2.0 / n) * diff.sum()
    db = db.reshape(1, 1)

    # 返回两个梯度。
    return dw, db


def mae_loss(y_hat, y):
    """
    平均绝对误差 MAE（Mean Absolute Error）。

        MAE = (1/n) * Σ |y_hat_i - y_i|

    特点：对异常值（outlier）不敏感，因为它不平方，误差大也只是线性增长。
    缺点：在误差为 0 的地方不可导（绝对值函数有个尖点），
         所以实现里我们约定"误差正好为 0 时梯度取 0"。
    """

    # 每个样本的误差取绝对值，再求平均。
    loss = np.abs(y_hat - y).mean()

    # 返回标量损失。
    return loss


def mae_grad(y_hat, y, x):
    """
    MAE 对 w、b 的梯度。

    推导：d|u|/du = sign(u)，其中 sign 是符号函数（正数返回 1，负数返回 -1）。

        ∂L/∂w = (1/n) * Σ sign(y_hat_i - y_i) * x_i
        ∂L/∂b = (1/n) * Σ sign(y_hat_i - y_i)

    注意这里没有系数 2（因为绝对值求导不会产生 2）。
    """

    # 这一批的样本数。
    n = x.shape[0]

    # 误差。
    diff = y_hat - y

    # np.sign 求符号：正数→1，负数→-1，零→0（这就顺手处理了不可导点）。
    sign = np.sign(diff)

    # 同样的矩阵乘法形式，注意系数是 1/n 而不是 2/n。
    dw = (1.0 / n) * (x.T @ sign)

    # 偏置的梯度。
    db = (1.0 / n) * sign.sum()
    db = db.reshape(1, 1)

    # 返回梯度。
    return dw, db


def huber_loss(y_hat, y, delta=1.0):
    """
    Huber 损失：MSE 和 MAE 的"折中版"。

    公式（对单个样本的误差 r = y_hat - y）：
        当 |r| <= delta 时： 0.5 * r^2          （小误差时像 MSE，处处光滑、收敛稳）
        当 |r| >  delta 时： delta * (|r| - 0.5 * delta)   （大误差时像 MAE，不被异常值带跑）

    参数:
        delta : 阈值，决定"多大的误差算大"。
    """

    # 每个样本的误差。
    r = y_hat - y

    # np.abs(r) 是每个误差的绝对值。
    abs_r = np.abs(r)

    # np.where(条件, 满足时的值, 不满足时的值)：numpy 版的"三元表达式"，
    # 它会逐元素判断，所以能一次算出所有样本的损失。
    per_sample = np.where(
        abs_r <= delta,                        # 条件：误差的绝对值不超过 delta
        0.5 * r ** 2,                          # 满足：用平方项
        delta * (abs_r - 0.5 * delta)          # 不满足：用线性项
    )

    # 最后对所有样本求平均，得到标量损失。
    return per_sample.mean()


def huber_grad(y_hat, y, x, delta=1.0):
    """
    Huber 损失对 w、b 的梯度。

    推导：
        当 |r| <= delta 时，dL/dr = r
        当 |r| >  delta 时，dL/dr = delta * sign(r)

    再乘上 ∂r/∂w = x、∂r/∂b = 1，并取平均即可。
    """

    # 这一批的样本数。
    n = x.shape[0]

    # 误差。
    r = y_hat - y

    # 同样用 np.where 逐元素决定"这次梯度用 r 还是用 delta*sign(r)"。
    # 这个数组可以理解为"每个样本的误差对损失的贡献系数"。
    dL_dr = np.where(np.abs(r) <= delta, r, delta * np.sign(r))

    # 对 w 的梯度：把每个样本的系数乘上它自己的 x_i 再平均。
    dw = (1.0 / n) * (x.T @ dL_dr)

    # 对 b 的梯度。
    db = (1.0 / n) * dL_dr.sum()
    db = db.reshape(1, 1)

    # 返回梯度。
    return dw, db


# =============================================================================
# 第三部分：一个"损失函数登记表"
# =============================================================================
# 把"损失函数"和"它对应的梯度函数"配成一对，方便训练脚本按名字选用。
# 这样 train.py 里只要写 loss_name="mse"，就能自动拿到配套的 loss 和 grad，
# 不用写一大堆 if/else。
LOSS_REGISTRY = {
    "mse":    (mse_loss,    mse_grad),      # 均方误差（默认，也是本任务重点）
    "mae":    (mae_loss,    mae_grad),      # 平均绝对误差
    "huber":  (huber_loss,  huber_grad),    # Huber（MSE 与 MAE 的折中）
}


# =============================================================================
# 第四部分：LinearRegression 类 —— 把参数 (w, b) 和模型操作打包在一起
# =============================================================================
class LinearRegression(object):
    """
    最简单的线性回归模型：y = w * x + b

    为什么用"类（class）"？
        因为模型有两样东西是天生绑在一起的：
            1) 参数 w、b（模型"学到"的知识）
            2) 用这些参数做预测的方法
        用类把它们装在一起，就像一个"小盒子"，
        换一份数据只要新建一个盒子，不会互相干扰。

    注意：这个类故意"很薄"——它只负责存参数和预测，
    真正的"怎么更新参数"由 train.py 里的优化器负责。
    这样做是为了让"模型"和"优化算法"这两件事彻底分开，
    这正是本任务想让我们理解的：模型 ≠ 训练算法。
    """

    def __init__(self, n_features=1):
        """
        构造函数：创建一个还没有训练的模型。

        参数:
            n_features : 特征个数。本任务只有 hours 一个特征，所以默认 1。
                         写成参数而不是写死 1，是为了以后加特征时不用改代码。
        """

        # 记录特征个数，后面检查形状时用得到。
        self.n_features = n_features

        # w（权重）初始化成形状 (n_features, 1) 的全零矩阵。
        # 形状写成二维是因为要和 x @ w 的矩阵乘法对上。
        self.w = np.zeros((n_features, 1), dtype=np.float64)

        # b（偏置）初始化成形状 (1, 1) 的零。
        # 注意：线性回归的损失是凸函数，参数全部初始化为 0 也没关系
        # （不像神经网络，全 0 初始化会导致对称性无法打破）。
        self.b = np.zeros((1, 1), dtype=np.float64)

        # 训练历史记录本，训练过程中把每轮的 loss 追加进来，方便最后画曲线。
        self.loss_history = []

    def predict(self, x):
        """
        用当前参数做预测。就是把 model.py 顶部的 forward 包一层，
        让调用者写成 model.predict(x)，读起来更自然。
        """

        # 直接复用 forward 函数，避免重复写公式。
        return forward(x, self.w, self.b)

    def compute_loss(self, x, y, loss_name="mse"):
        """
        计算当前参数在这份数据上的损失值，用于观察训练进展。
        """

        # 先算出预测值。
        y_hat = self.predict(x)

        # 从登记表里取出"损失函数"（元组的第 0 个元素）。
        loss_fn = LOSS_REGISTRY[loss_name][0]

        # 调用它算出损失并返回。
        return loss_fn(y_hat, y)

    def compute_grad(self, x, y, loss_name="mse"):
        """
        计算当前参数对应的梯度，用于更新参数。
        """

        # 先算出预测值。
        y_hat = self.predict(x)

        # 从登记表里取出"梯度函数"（元组的第 1 个元素）。
        grad_fn = LOSS_REGISTRY[loss_name][1]

        # 调用它，返回 (dw, db) 两个梯度。
        return grad_fn(y_hat, y, x)

    def get_params(self):
        """
        把参数取出来，方便外部打印或保存。
        返回的是 float 类型的标量（用 .item() 把 (1,1) 数组变成普通数字）。
        """

        # .item() 只能用在"只有一个元素"的数组上，正好符合 (1,1) 的形状。
        return self.w.item(), self.b.item()

    def set_params(self, w, b):
        """
        手动设置参数。用于：把闭式解的结果、或者某个实验的结果装回模型里。
        """

        # 把传入的数值重新整理成 (1,1) 的数组，保证形状一致。
        self.w = np.array([[float(w)]], dtype=np.float64)
        self.b = np.array([[float(b)]], dtype=np.float64)

    def __repr__(self):
        """
        定义"print(模型对象)"时显示什么。属于 Python 的"魔法方法"，
        目的是调试时能一眼看到参数，不用手动打印。
        """

        # 取出现有参数。
        w, b = self.get_params()

        # 拼成一句人能看懂的话，例如：LinearRegression(y = 6.1154 * x + 33.3054)
        return "LinearRegression(y = %.4f * x + %.4f)" % (w, b)


# =============================================================================
# 第五部分：闭式解（解析解）—— 用来验证梯度下降到底有没有算对
# =============================================================================
def fit_closed_form(x, y):
    """
    用最小二乘的"正规方程（Normal Equation）"直接算出最优参数。

    对于只有一个特征的情况，最优解有非常简洁的公式：

        w* = Σ (x_i - x̄)(y_i - ȳ) / Σ (x_i - x̄)^2
        b* = ȳ - w* * x̄

    其中 x̄ 是 x 的均值，ȳ 是 y 的均值。

    为什么要有这个函数？
        梯度下降是"一步一步逼近"最优解的迭代算法，它可能因为学习率没调好而没收敛。
        闭式解是"一步到位"的精确解（在只有一个特征时）。
        所以：
            - 报告里我们可以把它当作"标准答案"；
            - 它是检验我们梯度下降实现是否正确的"照妖镜"。
        这是一个非常实用的工程习惯：任何迭代算法，都尽量找一个能精确求解的小例子来对拍。

    参数:
        x : 特征，形状 (n, 1)
        y : 目标值，形状 (n, 1)

    返回:
        w : 最优权重（float）
        b : 最优偏置（float）
    """

    # 求特征的均值 x̄。
    x_mean = x.mean()

    # 求目标值的均值 ȳ。
    y_mean = y.mean()

    # 把 x、y 都"中心化"（减去各自均值），这样公式里就不用再写 - x̄、- ȳ。
    # 注意 None 这个参数的作用：把 x_mean 沿着"第 0 维"广播，
    # 保证无论 x 是什么形状，都做的是"每个元素减同一个均值"。
    x_centered = x - x_mean
    y_centered = y - y_mean

    # 分子：Σ (x_i - x̄)(y_i - ȳ)。两个 (n,1) 数组逐元素相乘再求和。
    numerator = (x_centered * y_centered).sum()

    # 分母：Σ (x_i - x̄)^2。
    denominator = (x_centered ** 2).sum()

    # 防御性编程：如果所有 x 都一样，分母是 0，w 无解（斜率可以是任意值）。
    # 这时直接抛出明确错误，比返回一个 nan 更容易排查。
    if denominator == 0.0:
        raise ValueError("所有特征的取值都相同，无法拟合出唯一的权重 w")

    # 计算最优权重。
    w = numerator / denominator

    # 计算最优偏置：直线必须经过点 (x̄, ȳ)。
    b = y_mean - w * x_mean

    # 注意返回的是标量，不是数组，用起来更省事。
    return float(w), float(b)


def mse_analytic(y_hat, y):
    """
    一个"只用 NumPy 一行"的 MSE 写法，用于和手写版本对拍。
    两个函数算出来的数必须完全一样，否则说明手写版有 bug。
    """

    # np.mean 对整个数组求平均，效果等价于 mse_loss 里的 (diff**2).mean()。
    return np.mean((y_hat - y) ** 2)


# =============================================================================
# 自测代码块：直接运行 `python model.py` 时执行
# =============================================================================
if __name__ == "__main__":

    # 用一组"自己造的小数据"来做单元测试。
    # 我们希望模型能恢复出 y = 3*x + 1 这条直线（数据就是按它造的）。
    x_test = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)   # 4 个样本，1 个特征
    y_test = np.array([[1.0], [4.0], [7.0], [10.0]], dtype=np.float64)  # 正好是 3*x + 1

    # ---------- 测试 1：闭式解能不能把真实参数找回来 ----------
    w_cf, b_cf = fit_closed_form(x_test, y_test)
    print("闭式解结果:      w = %.6f, b = %.6f  (期望 w=3, b=1)" % (w_cf, b_cf))

    # ---------- 测试 2：三种损失函数在一个已知例子上的取值 ----------
    y_hat_test = np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float64)  # 故意让预测全错
    print("MSE  = %.6f  (手算: (1+16+49+100)/4 = %.6f)"
          % (mse_loss(y_hat_test, y_test), (1 + 16 + 49 + 100) / 4.0))
    print("MAE  = %.6f  (手算: (|0-1|+|0-4|+|0-7|+|0-10|)/4 = 5.5)"
          % mae_loss(y_hat_test, y_test))
    print("Huber(delta=1) = %.6f" % huber_loss(y_hat_test, y_test, delta=1.0))
    print("手写 MSE 与 np.mean 版本是否一致:",
          abs(mse_loss(y_hat_test, y_test) - mse_analytic(y_hat_test, y_test)) < 1e-12)

    # ---------- 测试 3：梯度的"数值检查"（最硬核的一步） ----------
    # 思路：用"数值微分"来验证我们手推的梯度公式。做法是：
    #   把某个参数轻轻加一点点 eps，看损失增加了多少，
    #   用 (L(w+eps) - L(w-eps)) / (2*eps) 近似导数。
    #   如果它和我们解析求出的梯度非常接近，说明梯度推导没写错。
    # 这是深度学习实践中必做的 debug 手段，初学者也应该掌握。
    model = LinearRegression()                                # 新建一个模型（w=0, b=0）
    model.set_params(0.5, -0.3)                               # 故意设一组非零参数
    dw, db = model.compute_grad(x_test, y_test, loss_name="mse")   # 解析梯度

    eps = 1e-6                                                # 一个很小的扰动
    w0, b0 = model.get_params()                               # 记下原始参数
    model.set_params(w0 + eps, b0)                            # w 加一点
    loss_plus = model.compute_loss(x_test, y_test)            # 加后的损失
    model.set_params(w0 - eps, b0)                            # w 减一点
    loss_minus = model.compute_loss(x_test, y_test)           # 减后的损失
    dw_numeric = (loss_plus - loss_minus) / (2 * eps)         # 数值梯度

    model.set_params(w0, b0 + eps)                            # 对 b 做同样的事
    loss_plus = model.compute_loss(x_test, y_test)
    model.set_params(w0, b0 - eps)
    loss_minus = model.compute_loss(x_test, y_test)
    db_numeric = (loss_plus - loss_minus) / (2 * eps)

    # 打印两个结果对比。差的绝对值应该非常小（1e-9 量级）。
    print("梯度检查 dw: 解析=%.10f, 数值=%.10f, 差值=%.2e"
          % (dw.item(), dw_numeric, abs(dw.item() - dw_numeric)))
    print("梯度检查 db: 解析=%.10f, 数值=%.10f, 差值=%.2e"
          % (db.item(), db_numeric, abs(db.item() - db_numeric)))

    # ---------- 测试 4：类的基本功能是否正常 ----------
    model.set_params(w_cf, b_cf)                              # 把闭式解装进模型
    print("模型对象打印效果:", model)                          # 触发 __repr__
    print("用闭式解预测 x=1,2,3:", model.predict(x_test).reshape(-1).tolist())  # 应该接近 4, 7, 10
