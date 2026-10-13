# -*- coding: utf-8 -*-
# =============================================================================
# dataset.py  ——  数据加载与预处理模块
# -----------------------------------------------------------------------------
# 这个文件只干一件事：把硬盘上的 CSV 文件，变成"可以直接喂给模型的一堆数字矩阵"。
#
# 为什么要把"读数据"单独写一个文件？
#   因为机器学习项目里，数据部分和模型部分是两件独立的事：
#   换一份数据，模型代码可以完全不动；换一个模型，读数据的代码也可以完全不动。
#   这种"一个文件只负责一件事"的写法，叫模块化，是工程上最基本的好习惯。
#
# 本文件用到的数据格式（data/train.csv 和 data/test.csv）长这样：
#   hours,score
#   2.21,53.14
#   3.3,47.82
#   ...
#   第一行是"表头"（列的名字），第一列 hours 是输入特征，第二列 score 是我们要预测的目标。
# =============================================================================

# os 是 Python 自带的"操作系统"模块，这里只用它来拼路径、建目录。
import os

# csv 是 Python 自带的 CSV 文件读写模块，读逗号分隔的表格文件很方便。
import csv

# numpy 是 Python 做数值计算最常用的第三方库。
# 我们把列表（list）转成 numpy 数组（ndarray）后，才能做矩阵运算、批量加减乘除。
import numpy as np


# -----------------------------------------------------------------------------
# 第一步：解决 matplotlib 的缓存目录问题（一个真实踩到的坑）
# -----------------------------------------------------------------------------
# matplotlib 是画图库。它第一次运行时，会往系统用户目录写一个"字体缓存"。
# 但在一部分电脑/服务器上，那个目录没有写权限，就会打印一大串红色报错。
# 解决办法：提前把缓存目录改到项目自己的文件夹里（本项目已建好 .mplcache 文件夹）。
# 必须在 import matplotlib 之前设置这个环境变量，所以这段代码放在文件最上面。
_HERE = os.path.dirname(os.path.abspath(__file__))          # 当前文件所在目录，也就是 code/
_PROJECT_ROOT = os.path.dirname(_HERE)                      # 上一级目录，也就是项目根目录
_MPL_CACHE_DIR = os.path.join(_PROJECT_ROOT, ".mplcache")   # 缓存目录：项目根目录/.mplcache

# 目录可能不存在（比如别人只拷了 code/ 文件夹），所以先尝试建一下。
try:
    os.makedirs(_MPL_CACHE_DIR, exist_ok=True)              # exist_ok=True 表示"已存在也不报错"
    os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE_DIR)   # 告诉 matplotlib 用这个目录
except OSError:
    # 万一连这里也不能写（极端情况），就什么都不做，让 matplotlib 自己想办法，
    # 这样至少不会因为"缓存目录"这种小事让整个训练脚本崩溃。
    pass


# -----------------------------------------------------------------------------
# 常量：默认的数据文件位置
# -----------------------------------------------------------------------------
# 这两个路径指向项目里的 data/ 文件夹。
# 用 os.path.join 拼路径，而不是写 "data/train.csv"，
# 是因为 Windows 用反斜杠 \、Linux/macOS 用正斜杠 /，os.path.join 会自动选对。
TRAIN_CSV_PATH = os.path.join(_PROJECT_ROOT, "data", "train.csv")   # 训练集：只用来训练模型
TEST_CSV_PATH = os.path.join(_PROJECT_ROOT, "data", "test.csv")     # 测试集：只在最后评估一次


# =============================================================================
# 函数 1：load_csv —— 读一个 CSV 文件，把里面的数字变成两个 numpy 数组
# =============================================================================
def load_csv(file_path):
    """
    读取形如 "hours,score" 的两列 CSV 文件。

    参数:
        file_path : str, CSV 文件的路径。

    返回:
        x : numpy 数组，形状 (n,)，n 是样本个数，存的是输入特征（学习时长 hours）。
        y : numpy 数组，形状 (n,)，存的是目标值（成绩 score）。

    说明:
        这里故意"手写"读文件的逻辑，而不是用 pandas.read_csv 一行搞定，
        目的是让我们清楚看到数据到底是怎么进到程序里的。
    """

    # 准备两个空列表，用来一行一行地收集数据。
    x_list = []            # 收集所有的 hours
    y_list = []            # 收集所有的 score

    # open(..., "r", ...) 表示以"只读"方式打开文件。
    # encoding="utf-8" 指定编码，避免中文路径或中文内容出现乱码。
    # with 语句的好处：代码块结束后文件会自动关闭，不用手动写 f.close()。
    with open(file_path, "r", encoding="utf-8") as f:

        # csv.reader 会把文件的每一行切成一个字符串列表，例如 ["2.21", "53.14"]。
        reader = csv.reader(f)

        # 下一行代码读取"表头"这一行（hours,score），只是为了跳过它。
        # 如果不先读掉表头，下面 float("hours") 就会报错。
        header = next(reader)

        # 逐个遍历数据行。enumerate 只是为了在出错时能报出"第几行"。
        for line_no, row in enumerate(reader, start=2):     # start=2 是因为表头是第 1 行

            # 跳过空行（有的文件末尾会多一个换行，读出来是空列表 []）。
            if len(row) == 0:
                continue

            # 再加一层保护：如果这一行不满足"两列"的格式，就明确报错，
            # 而不是让程序在某个莫名其妙的地方崩掉。这对调试非常有帮助。
            if len(row) != 2:
                raise ValueError(
                    "文件 %s 第 %d 行格式不对，期望 2 列，实际 %d 列"
                    % (file_path, line_no, len(row))
                )

            # float(...) 把字符串 "2.21" 变成浮点数 2.21。
            # row[0] 是第一列（hours），row[1] 是第二列（score）。
            # .strip() 去掉数字两边可能存在的空格。
            x_list.append(float(row[0].strip()))
            y_list.append(float(row[1].strip()))

    # np.array(列表, dtype=np.float64) 把 Python 列表变成 numpy 数组。
    # 指定 float64（64 位浮点数）是为了后面计算精度稳定，不会出现"整数除法"之类的意外。
    x = np.array(x_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)

    # 一个简单的健壮性检查：如果文件是空的，后面算均值会得到 nan，不如现在就报错。
    if x.size == 0:
        raise ValueError("文件 %s 里没有读到任何数据，请检查文件内容" % file_path)

    # 同时返回 x 和 y，调用者可以用 x, y = load_csv(路径) 一次接住两个值。
    return x, y


# =============================================================================
# 函数 2：to_column —— 把一维数组变成"列向量"的二维形状 (n, 1)
# =============================================================================
def to_column(x):
    """
    把形状 (n,) 的一维数组，变成形状 (n, 1) 的二维数组。

    为什么需要这一步？
        因为在矩阵乘法里，"一列特征"必须写成 (样本数, 特征数) 的形状。
        100 个样本、1 个特征，就应该是 (100, 1)，而不是 (100,)。
        这是初学者最容易搞混的地方，所以我们统一在最开始就整理好形状。
    """

    # np.asarray(x) 保证 x 是 numpy 数组（如果已经是，就直接用，不复制数据）。
    x = np.asarray(x, dtype=np.float64)

    # reshape(-1, 1) 的意思是："行数你帮我算（-1），列数固定 1 列"。
    # 对 (100,) 调用它就得到 (100, 1)；对已经是 (100, 1) 的数组调用也没问题。
    return x.reshape(-1, 1)


# =============================================================================
# 函数 3：standardize —— 特征标准化（机器学习里非常关键的一步）
# =============================================================================
def standardize(x_train, x_test):
    """
    用"训练集的均值和标准差"，把训练集和测试集的特征都标准化。

    标准化的公式（对每一个特征减均值、除标准差）：
        x_std = (x - mean) / std

    做过之后，训练集特征的均值约等于 0，标准差约等于 1。

    为什么一定要做？—— 因为梯度下降的收敛速度，跟特征的"数量级"关系极大。
        hours 的取值范围大约是 1 ~ 10；
        而 score 的取值范围大约是 30 ~ 100。
        如果直接拿原始数据做梯度下降，参数 w 和 b 的尺度差很多，
        损失函数的等高线会变成一个又细又长的"峡谷"，
        梯度下降会在峡谷两壁之间来回震荡，收敛得非常慢，学习率稍大就发散。
        标准化之后，等高线接近一个"正圆"，梯度下降可以笔直地走向最低点。

    这里有两个必须注意的点：
        1) 必须用"训练集的" mean/std 去处理测试集。
           如果用测试集自己的统计量，就等于提前偷看了测试集的信息，
           这在机器学习里叫"数据泄露（data leakage）"，会让评估结果虚高。
        2) 函数把 mean/std 一起返回，因为最后我们要把预测结果"还原"回原始尺度。

    参数:
        x_train : 训练集特征，形状 (n_train,) 或 (n_train, 1)
        x_test  : 测试集特征，形状 (n_test,) 或 (n_test, 1)

    返回:
        x_train_std : 标准化后的训练集特征，形状 (n_train, 1)
        x_test_std  : 标准化后的测试集特征，形状 (n_test, 1)
        mean        : 训练集特征的均值（float）
        std         : 训练集特征的标准差（float）
    """

    # 先统一整理成列向量的形状，后面的计算就不用再操心形状问题了。
    x_train = to_column(x_train)
    x_test = to_column(x_test)

    # .mean() 求平均值。对 (n,1) 的数组调用它，得到一个标量（所有元素的均值）。
    mean = x_train.mean()

    # .std() 求标准差。标准差衡量数据"波动幅度"。
    std = x_train.std()

    # 防御性编程：如果某个特征所有取值都一样，标准差就是 0，
    # 除 0 会得到 inf 或 nan，整个训练就毁了。这时我们把它改成 1，相当于"不缩放"。
    if std == 0.0:
        std = 1.0

    # 对训练集做标准化。注意：是减训练集的 mean、除训练集的 std。
    x_train_std = (x_train - mean) / std

    # 对测试集做"同一套"变换（关键！），保证两个集合在同一个坐标系里。
    x_test_std = (x_test - mean) / std

    # 四个返回值，缺一不可。
    return x_train_std, x_test_std, mean, std


# =============================================================================
# 函数 4：restore_params —— 把"标准化空间"里的参数，翻译回"原始空间"
# =============================================================================
def restore_params(w_std, b_std, mean, std):
    """
    把在标准化数据上训练得到的参数 (w_std, b_std)，
    换算成"用原始 hours 直接计算"的参数 (w, b)。

    推导过程（这一步很能体现线性回归的数学功底）：

        标准化后的特征：  x_std = (x - mean) / std
        模型在标准化空间：  y = w_std * x_std + b_std
        把 x_std 代进去：   y = w_std * (x - mean) / std + b_std
        展开整理：          y = (w_std / std) * x + (b_std - w_std * mean / std)
        对比 y = w * x + b，就得到：

            w = w_std / std
            b = b_std - w_std * mean / std

    这么做的好处：最终打印出来的模型是 "score = 6.1 * hours + 33.3" 这种
    人能直接看懂的形式，而不是一个在标准化空间里的抽象数字。

    参数:
        w_std, b_std : 标准化空间里的权重和偏置（float）
        mean, std    : 标准化时用的均值和标准差（float）

    返回:
        w, b : 原始空间（原始 hours 单位）下的权重和偏置
    """

    # 就是上面推导出来的两个公式，一行一个。
    w = w_std / std
    b = b_std - w_std * mean / std

    # 返回换算后的参数。
    return w, b


# =============================================================================
# 函数 5：iterate_minibatches —— 小批量（mini-batch）数据生成器
# =============================================================================
def iterate_minibatches(x, y, batch_size, shuffle=True, rng=None):
    """
    把数据切成一个个"小批量"，每次 yield 一小批出去。

    这是"小批量梯度下降（Mini-batch Gradient Descent）"的数据来源。

    什么是 yield？
        普通函数用 return，一返回函数就结束了。
        带 yield 的函数叫"生成器（generator）"：每次被调用到 yield 时，
        它会把值交出去、"暂停"在那里；等下一次要值时，再从暂停处继续往下跑。
        这样我们就不用一次性把几百个小批量都存进内存。

    参数:
        x          : 特征，形状 (n, 1)
        y          : 目标值，形状 (n, 1)
        batch_size : 每个小批量装多少个样本。
                     如果 >= 样本总数，就相当于"批量梯度下降（BGD，全量）"。
        shuffle    : 是否在每个 epoch 开始时打乱顺序。
                     打乱很重要！否则每个 epoch 的批量组成完全一样，
                     模型容易形成"固定的学习顺序偏见"，SGD 尤其需要打乱。
        rng        : numpy 随机数生成器。传进来而不是在函数里新建，
                     是为了让整个实验的随机性可复现（同一颗种子 → 同样的结果）。

    产出（每次 yield）:
        xb, yb : 一小批特征和小一批目标值
    """

    # x 有多少行，就有多少个样本。x.shape[0] 取第 0 维的长度。
    n_samples = x.shape[0]

    # 索引数组 [0, 1, 2, ..., n-1]，它代表"所有样本的编号"。
    indices = np.arange(n_samples)

    # 需要打乱就打乱。
    if shuffle:
        # 如果调用者没给随机数生成器，就新建一个（这时候结果不可复现，仅作兜底）。
        if rng is None:
            rng = np.random.default_rng()
        # rng.permutation 返回一个"洗过牌"的索引数组，顺序随机但每个编号出现一次。
        indices = rng.permutation(indices)

    # 按 batch_size 的步长切分索引。
    # range(0, n, batch_size) 会生成 0, batch_size, 2*batch_size, ...
    for start in range(0, n_samples, batch_size):

        # 当前批量的索引：从 start 一直取到 start + batch_size（不含）。
        # 最后一批可能不够 batch_size 个，切片会自动截断，这是正常现象。
        batch_idx = indices[start:start + batch_size]

        # x[batch_idx] 是 numpy 的"花式索引"：
        # 它会把 batch_idx 里指定的那些行按顺序取出来，组成一个新的数组。
        xb = x[batch_idx]      # 形状 (当前批量大小, 1)
        yb = y[batch_idx]      # 形状 (当前批量大小, 1)

        # yield 把这一小批交出去，函数在这里"暂停"。
        yield xb, yb


# =============================================================================
# 函数 6：prepare_data —— 把上面这些步骤串成一条"流水线"
# =============================================================================
def prepare_data(train_path=TRAIN_CSV_PATH, test_path=TEST_CSV_PATH):
    """
    一次性完成：读训练集 → 读测试集 → 标准化 → 整理成 (n,1) 的矩阵。

    参数:
        train_path : 训练集 CSV 路径（默认项目里的 data/train.csv）
        test_path  : 测试集 CSV 路径（默认项目里的 data/test.csv）

    返回一个字典（dict），里面装好训练脚本需要的一切：
        x_train_std : 标准化后的训练特征，形状 (n_train, 1)
        y_train     : 训练集目标值，形状 (n_train, 1)
        x_test_std  : 标准化后的测试特征（用的是训练集的 mean/std），形状 (n_test, 1)
        y_test      : 测试集目标值，形状 (n_test, 1)
        x_train_raw : 原始训练特征（画图要用它当横坐标），形状 (n_train, 1)
        x_test_raw  : 原始测试特征，形状 (n_test, 1)
        mean, std   : 标准化参数，用于最后把参数还原回原始尺度
        n_train     : 训练样本数量（int）
        n_test      : 测试样本数量（int）
    """

    # 读训练集：训练集负责"学习参数"。
    x_train_raw, y_train_raw = load_csv(train_path)

    # 读测试集：测试集负责"最后考一次试"，训练全程不要偷看它。
    x_test_raw, y_test_raw = load_csv(test_path)

    # 用训练集的统计量做标准化，得到标准化后的特征和 mean/std。
    x_train_std, x_test_std, mean, std = standardize(x_train_raw, x_test_raw)

    # 用字典把结果一次性返回。比返回 8 个值更不容易接错顺序。
    data = {
        "x_train_std": x_train_std,                       # (n_train, 1)
        "y_train": to_column(y_train_raw),                # (n_train, 1)
        "x_test_std": x_test_std,                         # (n_test, 1)
        "y_test": to_column(y_test_raw),                  # (n_test, 1)
        "x_train_raw": to_column(x_train_raw),            # (n_train, 1) 原始尺度，画图用
        "x_test_raw": to_column(x_test_raw),              # (n_test, 1)  原始尺度，画图用
        "mean": mean,                                     # float
        "std": std,                                       # float
        "n_train": x_train_std.shape[0],                  # int，训练样本数
        "n_test": x_test_std.shape[0],                    # int，测试样本数
    }

    # 返回这条流水线的最终产物。
    return data


# =============================================================================
# 自测代码块：直接运行 `python dataset.py` 时会执行这里
# =============================================================================
# if __name__ == "__main__" 的意思是：
#   只有当这个文件被"直接运行"时才执行下面的代码；
#   当它被别的文件 import 时，下面的代码不会执行。
# 这是 Python 脚本的标准写法，方便我们单独测试这个模块。
if __name__ == "__main__":

    # 调用流水线函数，拿到数据。
    data = prepare_data()

    # 打印一些基本信息，确认数据读对了、形状符合预期。
    print("训练样本数:", data["n_train"])
    print("测试样本数:", data["n_test"])
    print("训练特征形状 (标准化后):", data["x_train_std"].shape)
    print("训练目标形状:", data["y_train"].shape)
    # 标准化之后，训练特征的均值应该非常接近 0，标准差应该非常接近 1。
    print("标准化后训练特征的均值: %.6f" % data["x_train_std"].mean())
    print("标准化后训练特征的标准差: %.6f" % data["x_train_std"].std())
    # 用训练集的 mean/std 去标准化测试集，所以测试集的均值不一定正好是 0，但应该很接近。
    print("标准化后测试特征的均值: %.6f" % data["x_test_std"].mean())
    # 打印前 5 个原始样本，肉眼核对一下和 CSV 文件里是不是一致。
    print("训练集前 5 个样本 (hours, score):")
    for i in range(5):
        print("   %.2f -> %.2f" % (data["x_train_raw"][i, 0], data["y_train"][i, 0]))

    # 顺便演示一下小批量生成器的效果：
    # 假设每批 32 个样本，看看第一批装了多少个、长什么形状。
    print("演示 mini-batch 切分（batch_size=32）:")
    for i, (xb, yb) in enumerate(iterate_minibatches(
            data["x_train_std"], data["y_train"], batch_size=32, rng=np.random.default_rng(0))):
        # %-8d 之类只是让打印出来的列对齐，看着舒服。
        print("   第 %d 批: xb.shape=%s, yb.shape=%s" % (i + 1, xb.shape, yb.shape))
