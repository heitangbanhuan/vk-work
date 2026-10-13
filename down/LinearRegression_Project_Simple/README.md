# 怎么运行这个项目

## 一、安装依赖

在项目文件夹下打开命令行，执行：

```
pip install -r requirements.txt
```

## 二、运行

代码里的路径是相对路径，所以**要先进入 code 文件夹**：

```
cd code
python train.py
```

`train.py` 会做这些事：
- 读 `data/train.csv` 和 `data/test.csv`
- 用梯度下降训练一个线性回归
- 试不同的学习率、不同的批量大小，把结果打印出来
- 在测试集上算 MSE / RMSE / R2
- 画三张图，保存到 `results/figure.png`
- 把关键数字保存到 `results/result.json`

然后生成报告：

```
python make_report.py
```

生成的 `report.pdf` 在项目根目录下。

## 三、想单独测试某个文件

`dataset.py` 和 `model.py` 都可以单独运行，会把中间结果打印出来：

```
python dataset.py     # 看数据读对了没有
```

## 四、文件都是干什么的

```
LinearRegression_Project_Simple/
├── code/
│   ├── dataset.py       读 csv、标准化
│   ├── model.py         线性回归的公式、损失、梯度
│   ├── train.py         训练 + 对比实验 + 画图
│   └── make_report.py   把结果排成 pdf
├── data/
│   ├── train.csv        训练集（100 条）
│   └── test.csv         测试集（30 条）
├── results/
│   ├── figure.png       三张实验结果图
│   └── result.json      跑出来的所有数字
├── report.pdf           实验报告
├── requirements.txt     需要装的库
└── note.md              学习笔记
```
