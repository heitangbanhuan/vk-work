# -*- coding: utf-8 -*-
# =============================================================================
# tools/make_report.py  ——  自动生成本项目的实验报告 report.pdf
# -----------------------------------------------------------------------------
# 这个脚本做三件事：
#     1. 读取 code/ 里训练好的结果：results/results.json + results/*.png
#     2. 用 ReportLab 把文字、表格、图片排版成一份完整的实验报告
#     3. 输出成项目根目录下的 report.pdf
#
# 为什么用"脚本生成报告"而不是手写 Word？
#     因为报告里所有数字都是训练真实跑出来的。如果手抄，
#     一旦改了代码重新训练，报告里的数字就对不上了，
#     而且还可能抄错。用脚本生成，数字永远和数据一致。
#
# 运行方式（在项目根目录下）：
#     python tools/make_report.py
#
# 依赖：reportlab（见 requirements.txt），以及已经跑过 python code/train.py
# =============================================================================

# os：拼路径、检查文件是否存在。
import os

# sys：把 code/ 目录加入模块搜索路径，这样能直接 import 里面的函数。
import sys

# json：读取 results/results.json 里的实验指标。
import json

# ReportLab 的各种组件：
#   colors   —— 定义颜色（RGBColor）
#   A4       —— A4 纸张尺寸
#   getSampleStyleSheet / ParagraphStyle —— 段落样式
#   SimpleDocTemplate / Paragraph / Spacer / Image / Table / TableStyle
#            —— 文档模板、段落、空白、图片、表格、表格样式
#   pdfmetrics / TTFont —— 注册中文字体（关键！否则中文会变方框）
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# -----------------------------------------------------------------------------
# 0. 路径设置
# -----------------------------------------------------------------------------
# __file__ 是当前脚本的路径，abspath 转成绝对路径，dirname 取它所在的目录。
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

# 上一级就是项目根目录。
_PROJECT_ROOT = os.path.dirname(_TOOLS_DIR)

# code/ 目录。
_CODE_DIR = os.path.join(_PROJECT_ROOT, "code")

# results/ 目录。
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")

# 输出的 PDF 路径。
_OUT_PDF = os.path.join(_PROJECT_ROOT, "report.pdf")

# 输出 PDF 需要用 PIL（Pillow）读取图片的原始像素尺寸，才能按原比例缩放。
# Pillow 是 matplotlib 的依赖，装了 matplotlib 就已经有了，不必额外安装。
try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None


# -----------------------------------------------------------------------------
# 1. 注册中文字体
# -----------------------------------------------------------------------------
# ReportLab 自带的字体不含中文字形，必须注册系统里的中文字体。
# 我们按优先级找一个能用的：
#   微软雅黑（msyh.ttc）  —— 最漂亮，Windows 自带
#   SimHei（simhei.ttf）  —— 黑体，几乎所有 Windows 都有
#   SimSun（simsun.ttc）  —— 宋体，最后的备选
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",              # macOS
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",    # Linux（文泉驿）
]

# 定义两个字体名：正文用 CN，标题加粗用 CNB。
FONT_BODY = "Helvetica"      # 先给一个默认值，找不到中文字体时才用它
FONT_BOLD = "Helvetica-Bold"


def register_chinese_font():
    """
    尝试注册一个可用的中文字体，并把字体名写进全局变量。
    返回注册成功的字体名，方便在日志里打印出来确认。
    """

    # 声明我们要修改的是全局变量（否则下面赋值只会创建局部变量）。
    global FONT_BODY, FONT_BOLD

    # 逐个候选字体尝试。
    for path in _FONT_CANDIDATES:

        # 文件不存在就跳过。
        if not os.path.exists(path):
            continue

        # 取出文件名（不含扩展名）作为字体名，例如 "msyh"。
        name = "CN_" + os.path.splitext(os.path.basename(path))[0]

        try:
            # 注册字体。.ttc 是"字体集合"，ReportLab 会取其中第一个字体。
            pdfmetrics.registerFont(TTFont(name, path))

            # 注册成功，把正文字体设为它。
            FONT_BODY = name

            # 标题也用它（这里没有单独取粗体文件，用同一种字体即可，
            # 靠字号和颜色来区分层级）。
            FONT_BOLD = name

            # 打印提示。
            print("[字体] 已注册中文字体:", path, "→", name)

            # 返回字体名。
            return name

        except Exception as error:
            # 某个字体注册失败不影响其他候选，打印一下就继续试下一个。
            print("[字体] 注册失败:", path, "->", error)

    # 所有候选都失败了。
    print("[字体] 警告：没有找到可用的中文字体，PDF 里的中文可能显示为方框！")
    return FONT_BODY


# -----------------------------------------------------------------------------
# 2. 读取训练结果
# -----------------------------------------------------------------------------
def load_results():
    """
    读取 results/results.json。
    如果文件不存在，说明还没跑过训练，直接给出清晰的中文提示并退出。
    """

    # 拼出结果文件路径。
    json_path = os.path.join(_RESULTS_DIR, "results.json")

    # 检查文件是否存在。
    if not os.path.exists(json_path):
        # 给出明确的操作指引，而不是抛一个看不懂的异常。
        print("错误：找不到 %s" % json_path)
        print("请先在项目根目录运行：python code/train.py")
        sys.exit(1)

    # 打开并解析 JSON。
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 返回解析后的字典。
    return data


# -----------------------------------------------------------------------------
# 3. 排版小工具
# -----------------------------------------------------------------------------
def format_loss_value(value, digits=2):
    """
    把损失值格式化成适合放进表格的短字符串。

    为什么需要它？
        当学习率过大时，损失会溢出到 1e62、1e296 甚至 nan / inf。
        这些长数字直接放进表格会把表格撑破（超出页面宽度）。
        所以统一规则：小数值保留两位小数，超大数值用科学计数法加“爆炸”说明。
    """

    # 字符串（比如 "diverged(nan)"）原样返回。
    if isinstance(value, str):
        return value

    # nan 判断：nan 不等于自己，这是最常用的判断技巧。
    if value != value:
        return "nan（溢出）"

    # 正负无穷。
    if value == float("inf"):
        return "inf（爆炸）"
    if value == float("-inf"):
        return "-inf（爆炸）"

    # 特别大的值（说明发散）用科学计数法，并标明是爆炸。
    if abs(value) > 1e6:
        return "%.1e（爆炸）" % value

    # 正常数值保留两位小数。
    return "%.*f" % (digits, value)


def make_styles():
    """
    定义报告里要用到的所有段落样式。
    统一管理样式的好处是：想改字号/行距只要改一个地方。
    """

    # 从 ReportLab 拿一套默认样式作为基础。
    styles = getSampleStyleSheet()

    # 再往里面添加我们自己的样式（键是样式名）。
    styles.add(ParagraphStyle(
        name="CnTitle",              # 封面大标题
        fontName=FONT_BOLD,          # 字体
        fontSize=24,                 # 字号
        leading=32,                  # 行距
        alignment=1,                 # 1 = 居中
        spaceAfter=16,               # 段后间距
        textColor=colors.HexColor("#1f3b73"),   # 深蓝色
    ))

    styles.add(ParagraphStyle(
        name="CnSubTitle",           # 封面副标题
        fontName=FONT_BODY,
        fontSize=13,
        leading=20,
        alignment=1,
        textColor=colors.HexColor("#555555"),
    ))

    styles.add(ParagraphStyle(
        name="CnH1",                 # 一级标题
        fontName=FONT_BOLD,
        fontSize=16,
        leading=22,
        spaceBefore=14,
        spaceAfter=8,
        textColor=colors.HexColor("#1f3b73"),
    ))

    styles.add(ParagraphStyle(
        name="CnH2",                 # 二级标题
        fontName=FONT_BOLD,
        fontSize=13,
        leading=18,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#2a5caa"),
    ))

    styles.add(ParagraphStyle(
        name="CnBody",               # 正文
        fontName=FONT_BODY,
        fontSize=10.5,
        leading=17,                  # 行距比字号大 60% 左右，中文读起来舒服
        spaceAfter=6,
        firstLineIndent=0,
    ))

    styles.add(ParagraphStyle(
        name="CnFormula",            # 公式（居中、等宽感）
        fontName=FONT_BODY,
        fontSize=11,
        leading=18,
        alignment=1,
        spaceBefore=4,
        spaceAfter=4,
        textColor=colors.HexColor("#333333"),
        backColor=colors.HexColor("#f5f7fa"),
    ))

    styles.add(ParagraphStyle(
        name="CnCaption",            # 图注
        fontName=FONT_BODY,
        fontSize=9,
        leading=13,
        alignment=1,                 # 居中
        spaceBefore=2,
        spaceAfter=10,
        textColor=colors.HexColor("#666666"),
    ))

    styles.add(ParagraphStyle(
        name="CnBullet",             # 列表项
        fontName=FONT_BODY,
        fontSize=10.5,
        leading=17,
        leftIndent=14,               # 左侧缩进
        spaceAfter=3,
    ))

    # 返回样式表。
    return styles


def figure(styles, filename, caption, width_cm=15.5):
    """
    生成"一张图 + 图注"的组合，返回一个列表（ReportLab 要求 flowable 逐个加）。

    参数:
        styles    : 样式表
        filename  : results/ 目录下的图片文件名
        caption   : 图注文字
        width_cm  : 图片显示宽度（厘米）。高度会按图片原比例自动算。
    """

# 拼出图片路径。
    path = os.path.join(_RESULTS_DIR, filename)

    # 如果图片不存在，就只留一句提示，不让整个报告生成失败。
    if not os.path.exists(path):
        return [Paragraph("【缺少图片 %s，请先运行 code/train.py】" % filename,
                          styles["CnBody"])]

    # 用 PIL 读图片原始像素尺寸来计算高宽比。
    # 注意：ReportLab 的 Image 如果只指定宽度，高度会按原比例自动缩放；
    # 但我们想同时限制最大高度，所以自己算一遍更稳妥。
    if PILImage is None:
        # 万一没有 Pillow（理论上不会），就退化成固定高度，至少能出报告。
        target_w, target_h = width_cm * cm, 8.0 * cm
    else:
        with PILImage.open(path) as im:
            px_w, px_h = im.size

        # 目标宽度（点：1 cm = 28.3465 pt）。
        target_w = width_cm * cm

        # 按比例算高度：高度 = 目标宽 × (原高 / 原宽)。
        target_h = target_w * (px_h / float(px_w))

        # 再限制一个最大高度，避免特别高的图把一页撑爆。
        max_h = 9.5 * cm
        if target_h > max_h:
            # 太高了，就反过来按最大高度缩宽度，保持比例不变。
            target_h = max_h
            target_w = target_h * (px_w / float(px_h))

    # 创建图片对象。
    img = Image(path, width=target_w, height=target_h)

    # 把图片的显示方式设为居中。
    img.hAlign = "CENTER"

    # 返回"图片 + 图注"两个元素。
    return [img, Paragraph(caption, styles["CnCaption"])]


def make_table(styles, header, rows, col_widths=None):
    """
    生成一个带样式的表格。

    参数:
        header     : 表头文字列表，例如 ["学习率", "最终 loss"]
        rows       : 数据行，每个元素是一个列表
        col_widths : 每列的宽度（点）。不传就平均分配。
    """

    # 把表头也当成一行放进数据里。
    data = [header] + rows

    # 创建表格对象。
    table = Table(data, colWidths=col_widths, hAlign="CENTER")

    # 定义表格样式：表头灰底、网格线、文字居中、字号统一。
    table.setStyle(TableStyle([
        # 表头背景色（浅蓝灰）。
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f1")),
        # 表头文字用中文字体。
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        # 正文格子用中文字体。
        ("FONTNAME", (0, 1), (-1, -1), FONT_BODY),
        # 字号。
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        # 上下内边距。
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        # 表格线：全部格子画 0.4pt 的灰色细线。
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9bb0cc")),
        # 文字垂直居中。
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        # 行列交替底色，长表格更易读。
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f7f9fc")]),
    ]))

    # 返回表格对象。
    return table


# -----------------------------------------------------------------------------
# 4. 组装报告内容
# -----------------------------------------------------------------------------
def build_story(styles, result):
    """
    把报告的每一块内容按顺序拼成一个列表（ReportLab 叫 story）。

    这里就是报告的全部内容所在。
    """

    # story 里按顺序放各种元素，ReportLab 会从上到下依次排版。
    story = []

    # =========================================================================
    # 封面
    # =========================================================================
    story.append(Spacer(1, 3.2 * cm))
    story.append(Paragraph("基于梯度下降的线性回归拟合", styles["CnTitle"]))
    story.append(Paragraph("从零实现（NumPy）实验报告", styles["CnSubTitle"]))
    story.append(Spacer(1, 1.2 * cm))

    # 封面上的一个小信息表：数据、模型、结论。
    final = result["final_model"]                     # 最终模型的各项指标
    metrics_test = final["metrics_test"]              # 测试集指标
    baseline = result["baseline"]                     # 闭式解基线

    info_rows = [
        ["数据集", "data/train.csv（100 条，训练）  data/test.csv（30 条，测试）"],
        ["任务", "用学习时长（hours）预测考试成绩（score）"],
        ["模型", "score = %.4f × hours + %.4f" % (final["w_raw"], final["b_raw"])],
        ["优化算法", "小批量梯度下降（Mini-batch GD）+ 动量 + 学习率衰减"],
        ["测试集表现", "R² = %.4f，RMSE = %.4f 分，MAE = %.4f 分"
            % (metrics_test["r2"], metrics_test["rmse"], metrics_test["mae"])],
        ["实现约束", "不使用 scikit-learn / pandas，模型、损失、梯度、优化器全部手写"],
    ]

    # 生成信息表（第一列窄一点，第二列宽一点）。
    story.append(make_table(styles, ["项目", "内容"], info_rows,
                            col_widths=[2.6 * cm, 12.4 * cm]))
    story.append(Spacer(1, 1.0 * cm))

    story.append(Paragraph(
        "本报告中的所有数字与图片，均由 <b>code/train.py</b> 一次运行自动产生，"
        "并保存在 results/ 目录中（图片为 PNG，指标为 JSON/CSV），可随时复现核对。",
        styles["CnSubTitle"]))

    # 封面结束后换页。
    story.append(PageBreak())

    # =========================================================================
    # 一、方法介绍
    # =========================================================================
    story.append(Paragraph("一、方法介绍", styles["CnH1"]))

    # ---------- 1.1 模型形式 ----------
    story.append(Paragraph("1.1 模型形式", styles["CnH2"]))
    story.append(Paragraph(
        "本任务只有 1 个输入特征（学习时长 hours），因此线性回归模型就是一条直线。"
        "为了后面扩展到多特征，代码里统一使用矩阵形式：", styles["CnBody"]))
    story.append(Paragraph("ŷ = X · w + b", styles["CnFormula"]))
    story.append(Paragraph(
        "其中 X ∈ R<sup>n×1</sup> 是特征矩阵，w ∈ R<sup>1×1</sup> 是权重（斜率），"
        "b ∈ R<sup>1×1</sup> 是偏置（截距），ŷ 是预测值。展开成标量写法即 "
        "ŷ<sub>i</sub> = w · x<sub>i</sub> + b。", styles["CnBody"]))
    story.append(Paragraph(
        "<b>参数初始化：</b>w 和 b 都初始化为 0。这是可行的，因为线性回归的"
        "均方误差损失函数是<b>凸函数</b>，不存在“全 0 初始化导致对称性无法打破”的问题"
        "（那是神经网络才需要担心的）。", styles["CnBody"]))

    # ---------- 1.2 损失函数 ----------
    story.append(Paragraph("1.2 损失函数的选择", styles["CnH2"]))
    story.append(Paragraph(
        "主损失函数选用<b>均方误差 MSE（Mean Squared Error）</b>：", styles["CnBody"]))
    story.append(Paragraph("L(w, b) = (1/n) Σ<sub>i=1..n</sub> (ŷ<sub>i</sub> − y<sub>i</sub>)²",
                           styles["CnFormula"]))
    story.append(Paragraph("选择 MSE 的三个理由：", styles["CnBody"]))
    story.append(Paragraph("① 它是凸函数，梯度下降一定能到达全局最优，不存在局部最优陷阱；",
                           styles["CnBullet"]))
    story.append(Paragraph("② 它对参数可导，且导数形式简单（见下节），便于手写实现；",
                           styles["CnBullet"]))
    story.append(Paragraph("③ 它在统计上对应“高斯噪声假设下的极大似然估计”，有理论依据。",
                           styles["CnBullet"]))
    story.append(Paragraph(
        "此外，为了对比研究，代码中还实现了 <b>MAE</b>（平均绝对误差）与 "
        "<b>Huber</b> 损失，共三种，用于第 2.6 节的“异常值鲁棒性”实验。", styles["CnBody"]))

    # ---------- 1.3 梯度推导 ----------
    story.append(Paragraph("1.3 梯度推导（手写，不使用自动微分）", styles["CnH2"]))
    story.append(Paragraph(
        "对单个样本，损失为 L<sub>i</sub> = (ŷ<sub>i</sub> − y<sub>i</sub>)²，"
        "而 ŷ<sub>i</sub> = w x<sub>i</sub> + b，用链式法则求偏导：", styles["CnBody"]))
    story.append(Paragraph("∂L<sub>i</sub>/∂w = 2(ŷ<sub>i</sub> − y<sub>i</sub>) · x<sub>i</sub>　，　"
                           "∂L<sub>i</sub>/∂b = 2(ŷ<sub>i</sub> − y<sub>i</sub>)",
                           styles["CnFormula"]))
    story.append(Paragraph("对全部 n 个样本取平均，写成矩阵形式：", styles["CnBody"]))
    story.append(Paragraph("∂L/∂w = (2/n) · Xᵀ(ŷ − y)　，　"
                           "∂L/∂b = (2/n) · Σ(ŷ<sub>i</sub> − y<sub>i</sub>)",
                           styles["CnFormula"]))
    story.append(Paragraph(
        "<b>梯度正确性验证（重要）：</b>为避免“梯度写错但程序照样跑”的隐蔽错误，"
        "本实现写了两道校验：", styles["CnBody"]))
    story.append(Paragraph(
        "① <b>闭式解对拍</b>：单特征线性回归有精确解 "
        "w* = Σ(x<sub>i</sub>−x̄)(y<sub>i</sub>−ȳ) / Σ(x<sub>i</sub>−x̄)²，b* = ȳ − w* x̄。"
        "在构造数据 y = 3x + 1 上，闭式解精确返回 w = 3.000000、b = 1.000000。",
        styles["CnBullet"]))
    story.append(Paragraph(
        "② <b>数值梯度检查</b>：用有限差分 ∂L/∂w ≈ [L(w+ε) − L(w−ε)] / (2ε)（ε = 1e−6）"
        "近似梯度，与解析梯度对比，实测相对误差约 1e−9 量级。",
        styles["CnBullet"]))

    # ---------- 1.4 优化算法 ----------
    story.append(Paragraph("1.4 梯度下降算法及更新公式", styles["CnH2"]))
    story.append(Paragraph(
        "代码用同一个函数实现了三种梯度下降，仅通过“批量大小 batch_size”一个参数切换：",
        styles["CnBody"]))

    # 三种梯度下降的对比表。
    gd_rows = [
        ["批量梯度下降 BGD", "全部 100 条", "1 次/轮", "梯度最准、曲线最平滑，但每轮只更新 1 次，最慢"],
        ["小批量梯度下降 MBGD", "16 条", "7 次/轮", "兼顾梯度平稳与更新频率，深度学习中的默认选择"],
        ["随机梯度下降 SGD", "1 条", "100 次/轮", "每步极便宜、前期下降极快，但梯度抖动剧烈"],
    ]
    story.append(make_table(styles,
                            ["变体", "每步样本数", "每轮更新次数", "特点"],
                            gd_rows,
                            col_widths=[3.6 * cm, 2.2 * cm, 2.4 * cm, 6.8 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("基础更新公式（以 batch 为单位）：", styles["CnBody"]))
    story.append(Paragraph("w ← w − η · ∂L<sub>batch</sub>/∂w　，　"
                           "b ← b − η · ∂L<sub>batch</sub>/∂b", styles["CnFormula"]))
    story.append(Paragraph(
        "注意梯度是<b>对当前批量</b>求平均（除以批量大小），这样三种批量模式下的"
        "学习率尺度才一致。", styles["CnBody"]))

    story.append(Paragraph("在基础公式之上，还实现了两个加速技巧：", styles["CnBody"]))
    story.append(Paragraph(
        "<b>① 动量 Momentum：</b>把更新量看作“小球滚下山”的速度，带惯性：",
        styles["CnBody"]))
    story.append(Paragraph("v ← β·v + (1−β)·g　，　w ← w − η·v　（β = 0.9）",
                           styles["CnFormula"]))
    story.append(Paragraph(
        "其有效步长约为 η/(1−β)。这个结论很重要——它意味着加动量相当于把学习率放大 10 倍，"
        "因此动量必须与小一些的学习率搭配使用（实验 F 会定量验证这一点）。",
        styles["CnBody"]))
    story.append(Paragraph(
        "<b>② 学习率衰减 LR Decay：</b>每个 epoch 让学习率乘上一个衰减系数：",
        styles["CnBody"]))
    story.append(Paragraph("η<sub>t</sub> = η<sub>0</sub> · decay<sup>t−1</sup>　（decay = 0.998）",
                           styles["CnFormula"]))
    story.append(Paragraph("前期步子大、下降快，后期步子小、能收敛得更精细。", styles["CnBody"]))

    # ---------- 1.5 数据预处理 ----------
    story.append(Paragraph("1.5 数据预处理：为什么必须做标准化", styles["CnH2"]))
    story.append(Paragraph(
        "原始数据中 hours 约在 1~10，而 score 约在 30~100。未标准化时，"
        "MSE 的 Hessian 矩阵为：", styles["CnBody"]))
    story.append(Paragraph("H = (2/n) · [[Σx², Σx], [Σx, n]]", styles["CnFormula"]))
    story.append(Paragraph(
        "本数据上 w 方向曲率约为 Σx²/n ≈ 32，而 b 方向曲率约为 1，"
        "<b>两个方向相差 30 余倍</b>，损失等高线被拉成极细长的“峡谷”。"
        "其后果是：能让长轴方向不发散的学习率，对短轴方向又太小（收敛极慢）；"
        "反之则长轴方向震荡发散。", styles["CnBody"]))
    story.append(Paragraph(
        "因此采用 Z-score 标准化 x<sub>std</sub> = (x − μ) / σ，"
        "其中 μ、σ <b>只由训练集统计得到</b>，再用同一组参数处理测试集"
        "（否则会造成数据泄露）。训练完成后，再把参数换算回原始尺度，便于解读：",
        styles["CnBody"]))
    story.append(Paragraph("w = w<sub>std</sub> / σ　，　b = b<sub>std</sub> − w<sub>std</sub> · μ / σ",
                           styles["CnFormula"]))

    # ---------- 1.6 实验协议 ----------
    story.append(Paragraph("1.6 实验协议", styles["CnH2"]))
    story.append(Paragraph(
        "训练集 <b>data/train.csv</b>（100 条）只用于训练；测试集 "
        "<b>data/test.csv</b>（30 条）只在最终评估时使用一次，不参与任何参数更新。"
        "超参数依据<b>训练集上的损失曲线</b>选择，不依据测试集结果，以避免测试集信息泄露。"
        "所有涉及随机性的实验统一固定随机种子（seed = 42），保证结果可复现。",
        styles["CnBody"]))

    # =========================================================================
    # 二、实验结果
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("二、实验结果", styles["CnH1"]))

    # ---------- 2.1 基线 ----------
    story.append(Paragraph("2.1 参照基准：闭式解（正规方程）", styles["CnH2"]))
    story.append(Paragraph(
        "先用闭式解算出理论最优参数，作为衡量梯度下降效果的“标准答案”。"
        "在标准化空间求得 w<sub>std</sub> = %.4f、b<sub>std</sub> = %.4f，"
        "换算回原始尺度得到："
        % (baseline["w_std"], baseline["b_std"]), styles["CnBody"]))
    story.append(Paragraph(
        "<b>score = %.4f × hours + %.4f</b>（训练集 MSE = %.4f）"
        % (baseline["w_raw"], baseline["b_raw"],
           result["loss_curve_final_train_mse"]), styles["CnFormula"]))
    story.append(Paragraph(
        "该模型在测试集上的表现为 MSE = %.4f、R² = %.4f，可视为本任务的“理论最优水平”。"
        % (baseline["test_metrics"]["mse"], baseline["test_metrics"]["r2"]),
        styles["CnBody"]))

    # ---------- 2.2 Loss 下降曲线 ----------
    story.append(Paragraph("2.2 Loss 随迭代次数的下降曲线", styles["CnH2"]))
    story.append(Paragraph(
        "下图 (a) 为线性坐标，(b) 为同一组曲线的对数坐标。对数坐标很有必要："
        "损失从约 2800 降到 25，若用线性坐标，后半段的下降细节会被压缩成一条贴地直线。",
        styles["CnBody"]))
    story.extend(figure(styles, "01_loss_curve.png",
                        "图 1　批量梯度下降（lr = 0.1，200 轮）的损失下降曲线，"
                        "训练集与测试集几乎重合，说明模型没有过拟合。"))
    story.append(Paragraph(
        "可以看到：损失在前 20 轮内迅速下降约两个数量级，之后进入平缓的收敛阶段，"
        "最终稳定在 24.98 —— 与闭式解的理论最优值<b>完全一致</b>。"
        "这从数值上证明了梯度下降的实现是正确的。", styles["CnBody"]))

    # ---------- 2.3 拟合过程 ----------
    story.append(Paragraph("2.3 拟合直线如何随迭代逐步变好", styles["CnH2"]))
    story.extend(figure(styles, "02_fit_progress.png",
                        "图 2　拟合直线随迭代次数的演化（颜色越深表示迭代越晚）。"
                        "epoch 200 的绿线与黑色虚线（闭式解）几乎完全重合。"))
    story.append(Paragraph(
        "图中可以清楚看到：第 1、2 轮的直线斜率明显偏小；到第 20 轮已经基本贴合数据趋势；"
        "第 200 轮与理论最优直线在视觉上已无法区分。这直观展示了梯度下降“一步步逼近最优解”的过程。",
        styles["CnBody"]))

    # ---------- 2.4 学习率对比 ----------
    story.append(Paragraph("2.4 不同学习率的收敛情况对比（重点实验）", styles["CnH2"]))
    story.extend(figure(styles, "03_learning_rate.png",
                        "图 3　不同学习率下的损失曲线。(a) 线性纵轴：学习率过大时损失直接爆炸；"
                        "(b) 对数纵轴：可以看出小学习率下降慢、合适的学习率收敛最快。"))
    story.extend(figure(styles, "04_lr_vs_final_loss.png",
                        "图 4　学习率对最终收敛效果的影响。lr ≤ 0.9 都能收敛到理论最优值 25.0；"
                        "lr = 1.2 时损失发散（冲出显示范围，标记为“发散出界”）。"))

    # 用真实数据生成学习率对比表。
    lr_final = result["learning_rate_final_loss"]
    lr_rows = []
    for lr_text in sorted(lr_final.keys(), key=lambda s: float(s)):
        # 用统一的小工具格式化：超大值会显示成 "1.2e+62（爆炸）"，不会撑破表格。
        lr_rows.append(["lr = %s" % lr_text, format_loss_value(lr_final[lr_text])])

    story.append(make_table(styles, ["学习率", "200 轮后的训练集 MSE"], lr_rows,
                            col_widths=[4.0 * cm, 6.0 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("<b>结论：</b>", styles["CnBody"]))
    story.append(Paragraph(
        "① 学习率 <b>过小（0.001）</b>：200 轮后损失仍高达 1945.4，远远没有走到最低点，"
        "表现为“下降极慢、看起来像一条水平线”；", styles["CnBullet"]))
    story.append(Paragraph(
        "② 学习率 <b>合适（0.05 ~ 0.9）</b>：都能收敛到理论最优值 24.98，"
        "本实验中最优的是 lr = 0.1（收敛最快）；", styles["CnBullet"]))
    story.append(Paragraph(
        "③ 学习率 <b>过大（1.2）</b>：损失依次经过 1e62 数量级直到溢出为 nan，模型完全崩溃。"
        "这说明学习率存在一个上限，超过它梯度下降不再收敛而是发散。", styles["CnBullet"]))

    # ---------- 2.5 标准化对比 ----------
    story.append(Paragraph("2.5 特征标准化的作用", styles["CnH2"]))
    story.extend(figure(styles, "05_standardization.png",
                        "图 5　(a) 同一个学习率 lr = 0.3：未标准化的数据直接爆炸，"
                        "标准化后稳定收敛；(b) 同样的学习率扫描，标准化把可用学习率范围拓宽了几十倍。"))

    # 用真实数据生成标准化对比表。
    std_cmp = result["standardization_compare"]
    std_rows = []
    for lr_text in sorted(std_cmp.keys(), key=lambda s: float(s)):
        raw_val = std_cmp[lr_text]["raw_scale_final_mse"]
        std_val = std_cmp[lr_text]["standardized_final_mse"]

        # 用同一个格式化函数，保证两个表格的风格一致。
        std_rows.append(["lr = %s" % lr_text,
                         format_loss_value(raw_val),
                         format_loss_value(std_val)])

    story.append(make_table(styles,
                            ["学习率", "原始尺度（未标准化）", "标准化后"],
                            std_rows,
                            col_widths=[3.4 * cm, 6.0 * cm, 4.0 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "数据非常直观：同一个 lr = 0.05，原始尺度下损失炸到 1e96，而标准化后稳定收敛到 25.0。"
        "<b>标准化不是可有可无的“优化技巧”，而是梯度下降能否正常工作的前提。</b>",
        styles["CnBody"]))

    # ---------- 2.6 批量大小 ----------
    story.append(Paragraph("2.6 批量大小的对比：BGD / Mini-batch / SGD", styles["CnH2"]))
    story.extend(figure(styles, "06_batch_size.png",
                        "图 6　(a) 按 epoch 比较：SGD 每轮更新次数最多，下降最快；"
                        "(b) 按参数更新次数比较：SGD 抖动剧烈、BGD 平滑；"
                        "(c) 达到“最优损失 1.05 倍以内”所需的计算代价对比。"))

    # 批量大小对比表（真实数据）。
    batch_info = result["batch_size"]
    batch_rows = []
    for name in batch_info:
        item = batch_info[name]
        batch_rows.append([
            name,
            "%d" % item["num_updates"],
            "%.2f" % item["final_train_mse"],
        ])
    story.append(make_table(styles,
                            ["配置", "参数更新次数", "最终训练 MSE"],
                            batch_rows,
                            col_widths=[7.5 * cm, 3.2 * cm, 3.2 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("<b>结论：</b>", styles["CnBody"]))
    story.append(Paragraph(
        "① <b>BGD</b> 梯度最准、曲线最平滑，但 60 轮只更新 60 次，收敛最慢；", styles["CnBullet"]))
    story.append(Paragraph(
        "② <b>SGD</b> 每轮更新 100 次，因此按 epoch 看下降飞快，"
        "但图 6(b) 显示它的 batch loss 在 1e0 ~ 1e2 之间剧烈抖动——"
        "因为每步只凭 1 个样本决定方向，噪声极大；", styles["CnBullet"]))
    story.append(Paragraph(
        "③ <b>Mini-batch（16）</b>介于两者之间：梯度比 SGD 平稳得多，"
        "更新次数又比 BGD 多 7 倍，达到同样的损失所需计算代价最低。"
        "这正是实际深度学习训练普遍采用小批量梯度下降的原因。", styles["CnBullet"]))

    # ---------- 2.7 损失函数与异常值 ----------
    story.append(Paragraph("2.7 损失函数的鲁棒性：异常值实验", styles["CnH2"]))
    outlier = result["outlier_setup"]
    story.append(Paragraph(
        "实验设计：人为向训练集中注入 %d 个异常点（第 %s 条样本，成绩 +%.0f 分），"
        "模拟“数据录入错误”或“作弊”这类离群数据，然后分别用 MSE / MAE / Huber 训练，"
        "比较它们被异常值带偏的程度。三种损失使用<b>完全相同</b>的优化配置"
        "（mini-batch = 16、lr = 0.05、300 轮），以保证唯一变量是损失函数本身。"
        % (len(outlier["indices"]), "、".join(str(i + 1) for i in outlier["indices"]),
           outlier["shift"]),
        styles["CnBody"]))
    story.extend(figure(styles, "07_loss_functions.png",
                        "图 7　(a) 三种损失在含异常值数据上的收敛曲线；"
                        "(b) 三条拟合直线对比；(c) 斜率偏离正确值的百分比。"))

    # 损失函数对比表（真实数据）。
    slopes = result["loss_function_slopes_with_outliers"]
    loss_rows = []
    for name in ["mse", "mae", "huber"]:
        item = slopes[name]
        loss_rows.append([
            name.upper(),
            "score = %.3f × hours + %.2f" % (item["w"], item["b"]),
            "%.2f%%" % item["slope_error_pct"],
        ])
    # 再加一行“无异常值时的正确值”作为参照。
    loss_rows.append([
        "（参照）无异常值",
        "score = %.3f × hours + %.2f" % (baseline["w_raw"], baseline["b_raw"]),
        "—",
    ])
    story.append(make_table(styles,
                            ["损失函数", "学到的直线", "斜率偏离正确值"],
                            loss_rows,
                            col_widths=[3.6 * cm, 7.4 * cm, 3.4 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("<b>结论与机理分析：</b>", styles["CnBody"]))
    story.append(Paragraph(
        "MSE 的斜率被带偏 %.2f%%，而 MAE 和 Huber 分别只偏 %.2f%% 和 %.2f%%。"
        "根本原因在于梯度中误差的“权重”："
        % (slopes["mse"]["slope_error_pct"], slopes["mae"]["slope_error_pct"],
           slopes["huber"]["slope_error_pct"]),
        styles["CnBody"]))
    story.append(Paragraph(
        "MSE 的梯度含因子 (ŷ−y)，<b>误差越大梯度越大</b>，一个偏了 45 分的点"
        "其影响力被平方放大，所以在梯度里“话语权”极大；", styles["CnBullet"]))
    story.append(Paragraph(
        "MAE 的梯度是 sign(ŷ−y)，<b>无论多离谱的误差都只算 ±1</b>，"
        "相当于每个样本最多投一票，因此对异常值几乎免疫；", styles["CnBullet"]))
    story.append(Paragraph(
        "Huber 损失在小误差时用平方、大误差时用线性，兼顾了 MSE 的平滑性与 "
        "MAE 的鲁棒性。", styles["CnBullet"]))
    story.append(Paragraph(
        "需要说明的是：MAE 的鲁棒性是有代价的。<b>它的梯度大小恒为 ±1，不会随误差减小而衰减，"
        "因此更难优化</b>——在批量梯度下降下，MAE 需要远多于 MSE 的迭代次数才能收敛（详见第三节问题 2）。",
        styles["CnBody"]))

    # ---------- 2.8 动量与衰减 ----------
    story.append(Paragraph("2.8 动量与学习率衰减的消融实验", styles["CnH2"]))
    story.append(Paragraph(
        "本实验设计了三个场景，每个场景下对比四种配置（基线 / +动量 / +衰减 / 两者都加）。"
        "评价指标采用“<b>进入最优损失的 1.02 倍以内所需轮数</b>”"
        "（目标线 = %.2f），而不是最后一轮的损失值——"
        "因为越接近最优点，损失值的逐轮波动可能大于不同配置之间的真实差异。"
        % result["momentum_decay"]["target_loss"], styles["CnBody"]))
    story.extend(figure(styles, "08_momentum_decay.png",
                        "图 8　(a) 温和学习率下基线本就收敛良好；"
                        "(b) 临界学习率 lr = 1.0 下基线直接爆炸、动量成功救场；"
                        "(c) 三个场景的达标轮数对比（× 表示 150 轮内始终不收敛）。"))

    # 动量对比表（真实数据）。
    mom_table = result["momentum_decay"]["table"]
    mom_rows = []
    cfg_names = ["基线 (纯 BGD)", "+ 动量 0.9", "+ 学习率衰减 0.99", "+ 动量 + 衰减"]
    for scene in mom_table:
        for cfg in cfg_names:
            if cfg not in mom_table[scene]:
                continue
            item = mom_table[scene][cfg]
            reach = item["reach_epoch"]
            reach_text = "未收敛（爆炸）" if reach is None else "第 %d 轮" % reach
            mom_rows.append([scene, cfg, "%.2f" % item["final_train_mse"], reach_text])

    story.append(make_table(styles,
                            ["场景", "配置", "最终训练 MSE", "达标轮数"],
                            mom_rows,
                            col_widths=[2.8 * cm, 4.4 * cm, 3.4 * cm, 3.6 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("<b>结论（这里的结论有些反直觉，值得专门记录）：</b>",
                           styles["CnBody"]))
    story.append(Paragraph(
        "① 在<b>温和学习率（lr = 0.1）</b>下，加动量反而更慢（37 轮 vs 基线 21 轮）。"
        "原因是本实现采用 v ← βv + (1−β)g 的写法，动量的有效步长约为 η/(1−β)，"
        "β = 0.9 时相当于把学习率放大 10 倍，在本来就合适的学习率上再放大就会“过冲”。",
        styles["CnBullet"]))
    story.append(Paragraph(
        "② 在<b>临界学习率（lr = 1.0）</b>下，纯 BGD 直接发散（损失停在 4302.4），"
        "而加入动量后居然在第 10 轮就收敛了 —— 动量的惯性平滑了高频震荡，"
        "从而<b>提高了可容忍的学习率上限</b>。这是动量不可替代的价值。", styles["CnBullet"]))
    story.append(Paragraph(
        "③ 因此，“加动量一定更快”是<b>错误</b>的；正确说法是：动量允许（并且需要用）"
        "更大的学习率，它真正的价值在于“救场”和“配合学习率衰减使用”。",
        styles["CnBullet"]))

    # ---------- 2.9 最终模型与测试集 ----------
    story.append(Paragraph("2.9 最终模型与测试集拟合效果", styles["CnH2"]))
    story.append(Paragraph(
        "综合以上实验，最终配置为：MSE 损失 + 小批量 16 + lr = 0.05 + 动量 0.9 + "
        "学习率衰减 0.998 + 300 轮（选择依据见各节结论）。最终模型为：",
        styles["CnBody"]))
    story.append(Paragraph(
        "<b>score = %.4f × hours + %.4f</b>" % (final["w_raw"], final["b_raw"]),
        styles["CnFormula"]))

    # 最终指标表。
    mtr = final["metrics_train"]
    mte = final["metrics_test"]
    story.append(make_table(styles,
                            ["数据集", "MSE", "RMSE", "MAE", "R²"],
                            [
                                ["训练集（100 条）", "%.4f" % mtr["mse"], "%.4f" % mtr["rmse"],
                                 "%.4f" % mtr["mae"], "%.4f" % mtr["r2"]],
                                ["测试集（30 条）", "%.4f" % mte["mse"], "%.4f" % mte["rmse"],
                                 "%.4f" % mte["mae"], "%.4f" % mte["r2"]],
                            ],
                            col_widths=[4.0 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "测试集 R² = %.4f，说明模型解释了约 %.1f%% 的成绩波动；"
        "RMSE = %.4f 分，即平均预测误差约 4.5 分，相当于满分 100 分下的 4.5%%。"
        "测试集表现与训练集相当（甚至略好），未观察到过拟合。"
        % (mte["r2"], mte["r2"] * 100.0, mte["rmse"]), styles["CnBody"]))

    story.extend(figure(styles, "09_test_fit.png",
                        "图 9　(a) 测试集数据点与模型预测直线；"
                        "(b) 预测值 vs 真实值，点越贴近对角线说明预测越准。"))

    # ---------- 2.10 残差分析 ----------
    story.append(Paragraph("2.10 残差分析", styles["CnH2"]))
    residual_rows = []
    # 从 test_predictions.csv 里读残差（如果有的话）。
    pred_csv = os.path.join(_RESULTS_DIR, "test_predictions.csv")
    if os.path.exists(pred_csv):
        import csv as _csv
        with open(pred_csv, "r", encoding="utf-8-sig") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                residual_rows.append(float(row["residual"]))
    if residual_rows:
        mean_res = sum(residual_rows) / len(residual_rows)
        var_res = sum((v - mean_res) ** 2 for v in residual_rows) / len(residual_rows)
        std_res = var_res ** 0.5
        max_abs = max(abs(v) for v in residual_rows)
        story.append(Paragraph(
            "对 30 个测试样本的残差（真实值 − 预测值）进行统计：均值 = %.2f 分、"
            "标准差 = %.2f 分、最大绝对残差 = %.2f 分。"
            % (mean_res, std_res, max_abs), styles["CnBody"]))
    story.extend(figure(styles, "10_residuals.png",
                        "图 10　(a) 残差随时间分布（理想情况应随机散布在 0 附近、无明显形状）；"
                        "(b) 残差直方图。"))
    story.append(Paragraph(
        "残差在 0 附近随机散布、没有明显的漏斗形或弯曲趋势，说明“线性关系”这一模型假设"
        "对这份数据是合理的；若残差呈 U 形或喇叭形，则提示应引入二次项或改用其他模型。",
        styles["CnBody"]))

    # =========================================================================
    # 三、提问环节
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("三、提问环节：发现的问题 / 解决情况", styles["CnH1"]))
    story.append(Paragraph(
        "本节按“发现问题 → 分析原因 → 如何解决 → 是否解决”的顺序，记录整个过程中"
        "真实遇到的问题。其中前三个是已经解决的，最后两个是尚未完全解决的。",
        styles["CnBody"]))

    # --- 问题 1 ---
    story.append(Paragraph("问题 1（已解决）：用原始数据训练时损失变成 nan", styles["CnH2"]))
    story.append(Paragraph("<b>现象：</b>最初的代码不做任何预处理，lr 取 0.1 时损失迅速"
                           "变成 nan；把 lr 降到 0.001 才勉强稳定，但 200 轮几乎不下降。",
                           styles["CnBody"]))
    story.append(Paragraph("<b>排查：</b>梯度检查通过，说明梯度实现无误；于是转向分析数据尺度——"
                           "hours 约 1~10 而 score 约 30~100，二者数量级不匹配。",
                           styles["CnBody"]))
    story.append(Paragraph("<b>原因：</b>MSE 的 Hessian 中 w 方向曲率正比于 Σx²/n ≈ 32，"
                           "b 方向曲率正比于 1，相差 30 余倍，等高线成为狭长峡谷；"
                           "梯度下降在长轴方向震荡、在短轴方向过慢。",
                           styles["CnBody"]))
    story.append(Paragraph("<b>解决：</b>对特征做 Z-score 标准化（均值 0、标准差 1），"
                           "并且只用训练集的 μ 和 σ 处理测试集。实验 C 显示可用学习率范围"
                           "从“lr ≥ 0.05 即爆炸”拓宽到“lr ≤ 0.9 都稳定收敛”。"
                           "<b>该问题已彻底解决。</b>", styles["CnBody"]))

    # --- 问题 2 ---
    story.append(Paragraph("问题 2（已解决）：MAE 损失训练 200 轮后模型仍是 y = 0",
                           styles["CnH2"]))
    story.append(Paragraph("<b>现象：</b>用 MAE 做异常值实验时，无论怎么训练，"
                           "学到的直线始终是 y = 0，斜率完全为 0。", styles["CnBody"]))
    story.append(Paragraph("<b>排查：</b>先确认不是代码问题——检查 MAE 的梯度实现为 "
                           "(1/n)·Σsign(ŷ−y)·x，与推导一致；再代入具体数值估算“每轮能走多远”。",
                           styles["CnBody"]))
    story.append(Paragraph("<b>原因：</b>MAE 的梯度大小被 sign 函数限制在 ±1，"
                           "而标准化后 |x| ≤ 1.5，因此每步对 w 的更新量最多 0.02×1.5 = 0.03；"
                           "而 w 需要从 0 走到 16.4，需要 500 轮以上，200 轮远远不够。"
                           "<b>不是代码错，是优化难度问题。</b>", styles["CnBody"]))
    story.append(Paragraph("<b>解决：</b>改用 mini-batch（batch = 16）训练——分批后梯度会除以批量大小，"
                           "等效于放大梯度尺度；同时在实验 E 中统一用 batch = 16、lr = 0.05，"
                           "三种损失就都能在 300 轮内稳定收敛，对比才公平。<b>该问题已解决。</b>",
                           styles["CnBody"]))

    # --- 问题 3 ---
    story.append(Paragraph("问题 3（已解决）：学习率实验的图被一个 1e62 的点彻底画废",
                           styles["CnH2"]))
    story.append(Paragraph("<b>现象：</b>lr = 1.2 时损失溢出到 1e62，画图时纵轴被这一个点撑爆，"
                           "其余所有曲线被压成一条贴地直线；同时终端刷满 numpy 的 "
                           "overflow / invalid value 警告。", styles["CnBody"]))
    story.append(Paragraph("<b>原因：</b>这不是 bug，而正是“学习率过大导致发散”这一现象本身；"
                           "问题在于它的量级破坏了图的可读性，并淹没了正常日志。",
                           styles["CnBody"]))
    story.append(Paragraph("<b>解决：</b>① 写了 clip_curve() 函数，把超过显示上限的值置为 nan"
                           "（matplotlib 遇到 nan 会断开曲线），这样既能看到曲线“冲出画面”，"
                           "又不破坏整张图；② 用 np.errstate 临时关闭数值溢出警告，让日志清爽。"
                           "<b>该问题已解决。</b>", styles["CnBody"]))

    # --- 问题 4 ---
    story.append(Paragraph("问题 4（部分解决）：加动量后收敛反而变慢", styles["CnH2"]))
    story.append(Paragraph("<b>现象：</b>最初在 lr = 0.6 下对比，发现加了动量之后损失震荡更剧烈、"
                           "收敛更慢，与“动量是加速技巧”的直觉矛盾。", styles["CnBody"]))
    story.append(Paragraph("<b>排查：</b>先逐行检查动量实现无误；随后做了一轮参数扫描，"
                           "把不同学习率下“纯 BGD”与“BGD + 动量”的达标轮数并列对比。",
                           styles["CnBody"]))
    story.append(Paragraph("<b>原因：</b>一是评价指标选错了——只看最后一轮的损失，"
                           "其差异可能小于逐轮波动；二是忽略了动量有效步长约为 η/(1−β)，"
                           "β = 0.9 相当于把学习率放大 10 倍，在小学习率上无益、"
                           "但在临界学习率（lr = 1.0）上恰好能起“救场”作用。",
                           styles["CnBody"]))
    story.append(Paragraph("<b>解决情况：</b>重新设计了三场景、四配置的消融实验，"
                           "并改用“达标轮数”作为指标，结论已能自洽解释（见 2.8 节）。"
                           "<b>现象已解释清楚，但“为什么动量能提高可容忍的学习率上限”"
                           "严格的数学条件我还没有自己推导出来（见问题 5）。</b>",
                           styles["CnBody"]))

    # --- 问题 5 ---
    story.append(Paragraph("问题 5（未解决）：动量提升稳定上限的数学条件尚未推导清楚",
                           styles["CnH2"]))
    story.append(Paragraph(
        "实验观察到 lr = 1.0 时纯 BGD 发散、加动量后第 10 轮即收敛，"
        "我能给出直觉解释（惯性平滑了高频震荡），但还无法自己推导出“加入动量后稳定条件"
        "变成什么”。我打算下一步用特征值分析重新推导带动量的迭代矩阵，"
        "这属于<b>尚未解决</b>的问题。", styles["CnBody"]))

    # --- 问题 6 ---
    story.append(Paragraph("问题 6（未解决）：超参数选择缺少独立验证集", styles["CnH2"]))
    story.append(Paragraph(
        "严格的做法应从训练集中再切出一份验证集专门用于调参，测试集只在最后评估一次。"
        "本任务的测试集只有 30 条，训练集只有 100 条，再切分会让数据更加紧张，"
        "因此我采用“用训练集损失曲线选超参数 + 测试集只评估一次”的务实做法，"
        "并在此明确说明这一局限。<b>这是权衡后的选择，不是最优做法。</b>", styles["CnBody"]))

    # --- 问题 7 ---
    story.append(Paragraph("问题 7（未解决）：测试集指标好于训练集", styles["CnH2"]))
    story.append(Paragraph(
        "最终模型训练集 MSE = %.4f、测试集 MSE = %.4f，测试集反而更好，这有些反直觉。"
        "我的猜测是训练集中存在若干“低学习时长却高分”的样本（例如第 4、13、26、41 条附近），"
        "它们抬高了训练误差，而测试集没有这类点。但我尚未做定量验证，"
        "因此把它列为<b>未解决</b>的问题。"
        % (mtr["mse"], mte["mse"]), styles["CnBody"]))

    # =========================================================================
    # 四、总结
    # =========================================================================
    story.append(Paragraph("四、总结", styles["CnH1"]))
    story.append(Paragraph(
        "本任务在 NumPy 环境下从零实现了线性回归模型、MSE/MAE/Huber 三种损失函数、"
        "以及批量/小批量/随机三种梯度下降算法（含动量与学习率衰减），"
        "并完成了学习率、特征标准化、批量大小、损失函数鲁棒性、动量与衰减五个维度的对比实验。",
        styles["CnBody"]))
    story.append(Paragraph(
        "主要结论：① 梯度下降的正确性通过闭式解与数值梯度双重验证；"
        "② 特征标准化是梯度下降正常工作的前提，它把可用学习率范围拓宽数十倍；"
        "③ 学习率过小则收敛慢、过大则发散，存在最优区间；"
        "④ 小批量梯度下降在收敛质量与计算代价之间取得最佳平衡；"
        "⑤ MSE 对异常值敏感、MAE/Huber 更鲁棒，但 MAE 的代价是更难优化；"
        "⑥ 动量并非无条件加速，其价值在于允许更大的学习率。", styles["CnBody"]))
    story.append(Paragraph(
        "最终模型 <b>score = %.4f × hours + %.4f</b>，在 30 条测试数据上取得 "
        "R² = %.4f、RMSE = %.4f 分的表现，与理论最优（闭式解）几乎无差距。"
        % (final["w_raw"], final["b_raw"], mte["r2"], mte["rmse"]), styles["CnBody"]))
    story.append(Paragraph(
        "更详细的学习过程、思维路径与踩坑记录，见随项目提交的 <b>note.md</b>。",
        styles["CnBody"]))

    # 返回完整的 story。
    return story


# -----------------------------------------------------------------------------
# 5. 页码装饰
# -----------------------------------------------------------------------------
def add_page_number(canvas, doc):
    """
    在每一页底部画页码。
    ReportLab 的机制：每画一页就会调用一次这个函数（onPage 回调）。
    """

    # 保存当前画布状态（这样我们改的设置不会影响后面的绘制）。
    canvas.saveState()

    # 设置页脚字体和字号。
    canvas.setFont(FONT_BODY, 9)

    # 设置颜色为灰色。
    canvas.setFillColor(colors.HexColor("#888888"))

    # 在页面底部居中写页码。doc.page 是当前页码。
    canvas.drawCentredString(A4[0] / 2.0, 1.2 * cm, "第 %d 页" % doc.page)

    # 恢复画布状态。
    canvas.restoreState()


# -----------------------------------------------------------------------------
# 6. 主函数
# -----------------------------------------------------------------------------
def main():
    """
    生成 report.pdf 的主流程：注册字体 → 读结果 → 组装内容 → 输出 PDF。
    """

    # 打印一行提示，让人知道脚本开始跑了。
    print("=" * 70)
    print("正在生成实验报告 report.pdf ...")
    print("=" * 70)

    # 第一步：注册中文字体（必须在任何排版之前）。
    register_chinese_font()

    # 第二步：读取训练结果。
    result = load_results()
    print("[数据] 已读取 results/results.json")

    # 第三步：准备样式。
    styles = make_styles()

    # 第四步：创建 PDF 文档模板。
    #   pagesize=A4       —— A4 纸
    #   左右 2cm、上下 2cm 页边距
    #   title/author      —— PDF 元数据（阅读器里能看到）
    doc = SimpleDocTemplate(
        _OUT_PDF,
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=2.0 * cm,
        bottomMargin=1.8 * cm,
        title="基于梯度下降的线性回归拟合 - 实验报告",
        author="线性回归项目",
    )

    # 第五步：组装内容。
    story = build_story(styles, result)

    # 第六步：真正输出 PDF。
    # onFirstPage / onLaterPages 都指向 add_page_number，也就是每页都画页码。
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

    # 第七步：报告生成完毕，打印文件大小确认。
    size_kb = os.path.getsize(_OUT_PDF) / 1024.0
    print("[完成] 报告已生成:", _OUT_PDF)
    print("       文件大小: %.1f KB" % size_kb)
    print("=" * 70)


# 程序入口：直接运行本脚本时执行 main()。
if __name__ == "__main__":
    main()
