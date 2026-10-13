# make_report.py
# 这个小程序用来把训练结果排成一份 pdf 报告。

import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                               Table, TableStyle)

# 让 pdf 能显示中文，要注册一个中文字体
pdfmetrics.registerFont(TTFont("hei", "C:/Windows/Fonts/simhei.ttf"))

# 读训练保存下来的结果
f = open("../results/result.json", "r", encoding="utf-8")
r = json.load(f)
f.close()

# 准备几种文字样式
title = ParagraphStyle("title", fontName="hei", fontSize=20, leading=28, alignment=1)
h1 = ParagraphStyle("h1", fontName="hei", fontSize=14, leading=22, spaceBefore=12, spaceAfter=6)
body = ParagraphStyle("body", fontName="hei", fontSize=10.5, leading=17, spaceAfter=5)
caption = ParagraphStyle("caption", fontName="hei", fontSize=9, leading=13,
                         alignment=1, textColor=colors.grey)

# 报告的内容按顺序放进这个列表
story = []

story.append(Paragraph("基于梯度下降的线性回归拟合", title))
story.append(Spacer(1, 0.5 * cm))

# ---------------------------- 一、方法介绍 ----------------------------
story.append(Paragraph("一、方法介绍", h1))

story.append(Paragraph("1. 模型形式", h1))
story.append(Paragraph("数据里 x 是学习时长 hours，y 是成绩 score，只有一个特征，"
                       "所以模型就是一条直线：y = w * x + b。"
                       "代码里为了以后方便加特征，写成矩阵形式 y = x @ w + b。"
                       "w 和 b 一开始都设为 0。", body))

story.append(Paragraph("2. 损失函数", h1))
story.append(Paragraph("用均方误差 MSE：把每个样本的预测值和真实值之差平方，再求平均，"
                       "写成公式就是 L = (1/n) * 求和 (y_hat - y)^2。"
                       "平方有两个作用：一是正负误差不会互相抵消，"
                       "二是错得越多的样本会被罚得越重。", body))

story.append(Paragraph("3. 梯度下降和更新公式", h1))
story.append(Paragraph("损失对 w 和 b 求导（用链式法则推出来的）：", body))
story.append(Paragraph("dL/dw = (2/n) * 求和 (y_hat - y) * x", body))
story.append(Paragraph("dL/db = (2/n) * 求和 (y_hat - y)", body))
story.append(Paragraph("然后让参数往梯度的反方向走一小步，反复迭代：", body))
story.append(Paragraph("w = w - lr * dw", body))
story.append(Paragraph("b = b - lr * db", body))
story.append(Paragraph("lr 是学习率，决定每一步走多大。太小要走很多步，太大就会直接飞出去。", body))

story.append(Paragraph("4. 批量大小的三种选择", h1))
story.append(Paragraph("代码里用“每一批取多少个样本”这一个参数，就实现了三种梯度下降："
                       "每批用全部 100 个样本叫批量梯度下降，算出来的梯度最准，"
                       "但一个 epoch 只能更新 1 次；每批只用 1 个样本叫随机梯度下降，"
                       "一个 epoch 能更新 100 次，但梯度很抖；每批 16 个是折中，"
                       "也是实际最常用的做法。", body))

story.append(Paragraph("5. 数据标准化", h1))
story.append(Paragraph("hours 大概在 1 到 10 之间，score 大概在 30 到 100 之间，"
                       "两个数差了一个数量级。不做处理的话，损失函数的等高线会被拉成"
                       "又细又长的峡谷，梯度下降在里面来回震荡，很难收敛。"
                       "所以先把 x 变成“平均值 0、标准差 1”的形式：x = (x - mean) / std。"
                       "注意测试集要用训练集的 mean 和 std，不然就等于提前偷看了测试集。", body))

story.append(Paragraph("6. 数据划分", h1))
story.append(Paragraph("训练集 train.csv（100 条）只用来训练模型，"
                       "测试集 test.csv（30 条）只在最后评估一次，"
                       "不参与任何一次参数更新。", body))

# ---------------------------- 二、实验结果 ----------------------------
story.append(Paragraph("二、实验结果", h1))

story.append(Paragraph("1. 和闭式解对拍", h1))
story.append(Paragraph(
    "一个特征的线性回归有一个精确解（闭式解），可以直接用公式算出来，"
    "不需要迭代。我用它来检验梯度下降对不对。"
    "注意模型是在标准化后的 x 上训练的，所以要先把参数换算回原始的小时才有意义：", body))

# 把两个解都换算回原始尺度
# 标准化时是 x标准 = (x - mean) / std，所以换算公式是 w = w标准 / std，b = b标准 - w标准 * mean / std
mean = r["mean"]
std = r["std"]

cf_w = r["closed_form"]["w"] / std
cf_b = r["closed_form"]["b"] - r["closed_form"]["w"] * mean / std

gd_w = r["gradient_descent"]["w"] / std
gd_b = r["gradient_descent"]["b"] - r["gradient_descent"]["w"] * mean / std

story.append(Paragraph(
    "闭式解：y = %.4f * hours + %.4f" % (cf_w, cf_b), body))
story.append(Paragraph(
    "梯度下降（lr=0.1，每批 100 个，200 轮）：y = %.4f * hours + %.4f" % (gd_w, gd_b), body))
story.append(Paragraph(
    "两个结果只差在小数点后很多位，说明梯度推导和代码都是对的。", body))

story.append(Paragraph("2. 损失下降曲线和不同学习率的对比", h1))
# 图片按原来的长宽比例缩放，宽度用 15 厘米
story.append(Image("../results/figure.png", width=15 * cm, height=15 * 600 / 2250 * cm))
story.append(Paragraph("图 1　左：Loss 随 epoch 下降；中：不同学习率的对比；"
                       "右：测试集上的拟合效果。", caption))

# 学习率结果表
lr_rows = [["学习率", "200 轮后的训练损失"]]
for key in r["lr_result"]:
    value = r["lr_result"][key]
    # 1.2 的时候损失会涨到 1e62 那么大，写成一长串数字不好看，用科学计数法
    if value > 10000:
        text = "%.1e（已经发散）" % value
    else:
        text = "%.2f" % value
    lr_rows.append(["lr = " + key, text])

story.append(Spacer(1, 0.3 * cm))
table = Table(lr_rows, colWidths=[4 * cm, 6 * cm])
table.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, -1), "hei"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
]))
story.append(table)
story.append(Spacer(1, 0.3 * cm))

story.append(Paragraph(
    "从表里能看出三件事：学习率 0.001 太小了，200 轮之后损失还有 1945，"
    "离最低点差得远；学习率 0.05 到 0.9 都能收敛到 24.98（这就是理论最低值）；"
    "学习率 1.2 太大，损失直接涨到 1e62 然后变成 inf，模型完全废掉。", body))

story.append(Paragraph("3. 不同批量大小的对比", h1))

size_rows = [["每批样本数", "每个 epoch 更新次数", "200 轮后的损失"]]
for key in r["size_result"]:
    size = int(key)
    size_rows.append([key, str(100 // size), "%.4f" % r["size_result"][key]])

table2 = Table(size_rows, colWidths=[4 * cm, 4.5 * cm, 4.5 * cm])
table2.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, -1), "hei"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
]))
story.append(table2)
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "每批 100 个的时候损失最低（24.9786），因为每次更新用的信息最全。"
    "每批 1 个的时候损失最差（28.7995），因为梯度只由一个样本决定，抖得厉害，"
    "所以到达最低点附近之后还会被单个样本带着来回跳。"
    "每批 16 个介于两者之间，这就是小批量梯度下降被广泛使用的原因。", body))

story.append(Paragraph("4. 测试集上的拟合效果", h1))
story.append(Paragraph("最终模型：y = %.4f * hours + %.4f" % (gd_w, gd_b), body))
story.append(Paragraph("在训练集上的损失是 %.4f，在 30 条测试数据上：MSE = %.4f，"
                       "RMSE = %.4f 分，R2 = %.4f"
                       % (r["train_loss"], r["test_mse"], r["test_rmse"], r["test_r2"]), body))
story.append(Paragraph(
    "R2 = %.4f 表示模型解释了大约 %.1f%% 的成绩变化，"
    "RMSE = %.4f 分表示平均预测误差大概 4.5 分。"
    "测试集的损失和训练集很接近，说明模型没有过拟合。"
    "图 1 右边可以看到，测试集的点基本都落在直线附近。"
    % (r["test_r2"], r["test_r2"] * 100, r["test_rmse"]), body))

# ---------------------------- 三、提问环节 ----------------------------
story.append(Paragraph("三、提问环节", h1))

story.append(Paragraph("问题 1：为什么一定要做标准化？", h1))
story.append(Paragraph(
    "一开始我没做标准化，直接用原始的 hours 训练，学习率取 0.1 的时候损失很快就变成 nan。"
    "我以为是梯度写错了，但用数值微分检查过梯度之后发现没错。"
    "后来才想明白是数据量级的问题：hours 是 1 到 10，score 是 30 到 100，"
    "损失函数的等高线是一个很扁的椭圆，w 方向和 b 方向的曲率差了三十多倍，"
    "梯度下降在长轴方向容易震荡、在短轴方向又走得太慢。"
    "做了标准化之后，两个方向的尺度接近了，同一个学习率就能稳定收敛。", body))

story.append(Paragraph("问题 2：学习率为什么不能太大？", h1))
story.append(Paragraph(
    "实验里 lr = 0.9 还能收敛到最低点，lr = 1.2 就直接发散到 1e62。"
    "我的理解是：梯度下降每一步走 lr 乘梯度那么远，"
    "如果一步跨过了最低点、而且跨得比到最低点的距离还远，"
    "下一次就会跑到更远的地方，误差被越放越大，最后就飞出去了。", body))

story.append(Paragraph("问题 3：小批量为什么损失更高？", h1))
story.append(Paragraph(
    "每批只用 1 个样本的时候，最后损失是 28.80，比每批 100 个的 24.98 差不少。"
    "我一开始以为是小批量效果更差，后来想明白了：不是它学不到最优点，"
    "而是它每一步只看一个样本，方向很偏，走到最优点附近之后还会被单个样本带着来回跳，"
    "所以最后停下来的位置不一定是最好那个点。",
    body))

story.append(Paragraph("问题 4：图被一个特别大的数毁掉了", h1))
story.append(Paragraph(
    "第一次画“不同学习率对比”这张图的时候，图完全不能看：lr = 1.2 的损失是 1e62，"
    "这一个点把纵轴撑到了 1e62，其他几条正常曲线全被压成贴着横轴的一条直线。"
    "我试过的办法是把纵轴改成对数坐标，但 1e62 和 20 差得还是太多，"
    "最后只能在画图的时候把 lr = 1.2 这条线去掉。"
    "它的结果我保留在控制台输出和表里，这样信息没有丢，图也能看。", body))

story.append(Paragraph("还没解决的问题", h1))
story.append(Paragraph(
    "① 我是用训练集的损失来选学习率的，严格来说应该再切一份验证集专门调参，"
    "但数据只有 100 条，再切就更少了，所以这次没有做。"
    "② 画图那个问题其实没有真正解决，我只是把发散的那条线“藏起来”了。"
    "我还不知道怎样在一张图里既能看清正常曲线、又能看清爆炸的那条。", body))

# 生成 pdf
doc = SimpleDocTemplate("../report.pdf", pagesize=A4,
                        leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=2 * cm, bottomMargin=2 * cm)
doc.build(story)
print("报告已生成: report.pdf")
