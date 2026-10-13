# -*- coding: utf-8 -*-
"""重新生成 Python_Traffic_Light_Project/report.pdf（内容与 main.py / note.md 对齐）"""

import shutil

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

FONT_BODY = r"C:\Windows\Fonts\Deng.ttf"
FONT_HEI = r"C:\Windows\Fonts\simhei.ttf"
SRC_PDF = r"D:\vk\Python_Traffic_Light_Project\report.pdf"
OUT_PDF = r"D:\vk\Python_Traffic_Light_Project\report.pdf"
BACKUP_PDF = r"D:\vk\tmp\pdfs\report_original.pdf"

pdfmetrics.registerFont(TTFont("Deng", FONT_BODY))
pdfmetrics.registerFont(TTFont("SimHei", FONT_HEI))

PAGE_W, PAGE_H = A4
LEFT = RIGHT = 2.0 * cm
TOP = 1.8 * cm
BOTTOM = 1.8 * cm

ACCENT = colors.HexColor("#1F3864")
GRAY = colors.HexColor("#666666")
CODE_BG = colors.HexColor("#F4F4F4")
CODE_BORDER = colors.HexColor("#C8C8C8")


title_style = ParagraphStyle(
    "Title", fontName="SimHei", fontSize=20, leading=27,
    alignment=TA_CENTER, textColor=ACCENT, spaceAfter=4,
)
subtitle_style = ParagraphStyle(
    "Subtitle", fontName="Deng", fontSize=10.5, leading=15,
    alignment=TA_CENTER, textColor=GRAY, spaceAfter=2,
)
h1_style = ParagraphStyle(
    "H1", fontName="SimHei", fontSize=14, leading=19,
    textColor=ACCENT, spaceBefore=14, spaceAfter=6, keepWithNext=1,
)
h2_style = ParagraphStyle(
    "H2", fontName="SimHei", fontSize=12, leading=17,
    textColor=ACCENT, spaceBefore=10, spaceAfter=4, keepWithNext=1,
)
body_style = ParagraphStyle(
    "Body", fontName="Deng", fontSize=10.5, leading=16,
    firstLineIndent=21, spaceAfter=4,
)
intro_style = ParagraphStyle(
    "Intro", parent=body_style, keepWithNext=1,
)
bullet_style = ParagraphStyle(
    "Bullet", fontName="Deng", fontSize=10.5, leading=16,
    leftIndent=16, bulletIndent=2, spaceAfter=3,
)
code_style = ParagraphStyle(
    "Code", fontName="Deng", fontSize=9.3, leading=13.6,
    textColor=colors.HexColor("#222222"),
)
summary_style = ParagraphStyle(
    "Summary", fontName="Deng", fontSize=10.5, leading=16,
    firstLineIndent=21,
)


def h1(text):
    return Paragraph(text, h1_style)


def h2(text):
    return Paragraph(text, h2_style)


def body(text):
    return Paragraph(text, body_style)


def intro(text):
    return Paragraph(text, intro_style)


def bullet(text):
    return Paragraph(text, bullet_style, bulletText="•")


def code_block(text):
    cell = Preformatted(text, code_style)
    table = Table(
        [[cell]],
        colWidths=[PAGE_W - LEFT - RIGHT],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("BOX", (0, 0), (-1, -1), 0.8, CODE_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CCCCCC"))
    canvas.setLineWidth(0.6)
    canvas.line(LEFT, BOTTOM - 0.5 * cm, PAGE_W - RIGHT, BOTTOM - 0.5 * cm)
    canvas.setFont("Deng", 9)
    canvas.setFillColor(GRAY)
    canvas.drawCentredString(PAGE_W / 2.0, BOTTOM - 1.1 * cm, "第 %d 页" % doc.page)
    canvas.restoreState()


story = []

# ============ 标题 ============
story.append(Paragraph("十字路口交通灯控制系统实验报告", title_style))
story.append(Paragraph("Python 模拟 | main.py | 2026-09", subtitle_style))
story.append(Spacer(1, 6))

# ============ 1 方法介绍 ============
story.append(h1("1 方法介绍"))

story.append(h2("1.1 设计的交通灯状态"))
story.append(
    body(
        "系统采用“状态 + 持续时间 + 状态切换”的思路来模拟真实交通灯，共设计 4 个状态，"
        "每个状态对应南北、东西两个方向的灯色，并有固定的持续时长："
    )
)
story.append(bullet("状态 1：南北绿灯、东西红灯，持续 5 秒"))
story.append(bullet("状态 2：南北黄灯、东西红灯，持续 2 秒"))
story.append(bullet("状态 3：南北红灯、东西绿灯，持续 5 秒"))
story.append(bullet("状态 4：南北红灯、东西黄灯，持续 2 秒"))
story.append(
    body(
        "程序中状态用整数 1~4 表示，各状态对应的显示文本存放在字典 state_text 中；"
        "状态编号与持续时长的关系由函数 duration(state) 统一处理（状态 1、3 返回 5 秒，"
        "状态 2、4 返回 2 秒）。把“状态编号”和“显示文本”分开存放，做状态切换和倒计时时"
        "只操作数字，输出时再查表换成文字，代码更清晰。"
    )
)

story.append(h2("1.2 不同状态之间如何切换"))
story.append(
    intro(
        "正常情况下，状态按 1 → 2 → 3 → 4 → 1 的顺序不断循环，切换关系如下："
    )
)
story.append(
    code_block(
        "状态1（南北绿灯/东西红灯，5秒）\n"
        "  ↓\n"
        "状态2（南北黄灯/东西红灯，2秒）\n"
        "  ↓\n"
        "状态3（南北红灯/东西绿灯，5秒）\n"
        "  ↓\n"
        "状态4（南北红灯/东西黄灯，2秒）\n"
        "  ↓\n"
        "状态1 → 循环"
    )
)
story.append(
    body(
        "每个状态先显示自己的持续时长（每秒打印一次剩余时间），时间结束后调用状态切换函数"
        "得到下一个状态的编号，再进入下一轮循环，从而模拟真实交通灯控制器的连续运行。"
    )
)

story.append(h2("1.3 change_state() 函数的作用"))
story.append(
    intro(
        "题目要求设计一个 change_state(state) 函数，作用是根据当前状态决定下一个状态。"
        "main.py 与 任务1.py 中的实现即为 change_state(state)，返回值为下一个状态的编号："
    )
)
story.append(
    code_block(
        "def change_state(state):    # 根据当前状态返回下一个状态：1→2→3→4→1\n"
        "    return state % 4 + 1"
    )
)
story.append(
    body(
        "函数体只有一行，利用取模运算实现循环：change_state(1)=2、change_state(2)=3、"
        "change_state(3)=4，而 change_state(4) 时 4 % 4 + 1 = 1，正好回到状态 1。"
        "相比用多条 if/elif 逐个判断，"
        "这种方式代码更简洁，也不容易漏掉边界情况。"
    )
)

story.append(h2("1.4 while 循环在程序中的作用"))
story.append(
    body(
        "程序中用了几处循环，各司其职："
    )
)
story.append(
    bullet(
        "外层 while True：让主程序永不退出，模拟真实交通灯控制器一直运行，是整个系统的“总开关”；"
    )
)
story.append(
    bullet(
        "内层倒计时循环 for remain in range(duration(state), 0, -1)：控制当前状态持续规定秒数，"
        "每秒打印一次剩余时间；"
    )
)
story.append(
    bullet(
        "紧急模式内部 while True：在紧急状态下保持南北、东西全红，并持续检查是否按下了 0。"
    )
)
story.append(
    body(
        "其中外层 while True 的作用最关键：没有它，run() 执行一遍就结束了，交通灯无法自动"
        "连续循环，所以系统必须依赖这个永不退出的循环来模拟真实控制器。"
    )
)

story.append(h2("1.5 紧急模式是如何加入正常控制流程的"))
story.append(
    body(
        "紧急模式要求“无论当前处于什么状态，检测到 1 都立即进入全红，按 0 结束并回到状态 1”。"
        "加入时主要解决了“如何在不卡住倒计时的前提下检测按键”这个问题："
    )
)
story.append(
    bullet(
        "最初使用 input() 检测输入，但 input() 会阻塞程序，倒计时和状态切换全部停住；"
    )
)
story.append(
    bullet(
        "尝试过 threading 后台线程方案，但输入 1 之后还需要再按回车，交互不够方便；"
    )
)
story.append(
    bullet(
        "最终选用 Windows 下的 msvcrt 库：msvcrt.kbhit() 非阻塞地检查有没有按键，"
        "msvcrt.getch() 读取按键，不需要按回车。"
    )
)
story.append(
    body(
        "具体实现上，把每 1 秒拆成 20 个 0.05 秒的小段，每段调用一次 check_key() 检测按键；"
        "一旦检测到 1，就把 emergency_mode 置为 True，并立即 break 当前的倒计时。主循环开头判断 "
        "emergency_mode，为真时进入紧急模式分支：打印“南北：红灯 / 东西：红灯”，然后在内部"
        "循环中等待按 0；按 0 后把 state 重置为 1、emergency_mode 置回 False，系统从状态 1 "
        "重新开始正常循环。等待按 0 的循环里还加了 time.sleep(0.1)，避免程序空转占用 CPU。"
    )
)

# ============ 2 运行结果 ============
story.append(h1("2 运行结果"))
story.append(
    body(
        "运行环境：Windows 11，Python 3.14。直接执行 python main.py 得到以下结果"
        "（按实际运行顺序整理）。"
    )
)

story.append(h2("2.1 正常交通灯循环运行结果"))
story.append(
    intro(
        "系统启动后从状态 1 开始，自动倒计时并切换状态，持续循环。以下是正常运行时的输出："
    )
)
story.append(
    code_block(
        "交通灯控制系统启动\n"
        "提示：按 1 触发紧急模式，按 0 结束紧急模式\n"
        "当前状态：南北绿灯,东西红灯，持续 5 秒\n"
        "剩余时间：5 秒\n"
        "剩余时间：4 秒\n"
        "剩余时间：3 秒\n"
        "剩余时间：2 秒\n"
        "剩余时间：1 秒\n"
        "切换交通灯...\n"
        "当前状态：南北黄灯,东西红灯，持续 2 秒\n"
        "剩余时间：2 秒\n"
        "剩余时间：1 秒\n"
        "切换交通灯...\n"
        "当前状态：南北红灯,东西绿灯，持续 5 秒\n"
        "剩余时间：5 秒\n"
        "剩余时间：4 秒\n"
        "剩余时间：3 秒\n"
        "..."
    )
)
story.append(
    body(
        "可以看到绿灯状态倒计时 5 秒、黄灯状态倒计时 2 秒，与题目规定一致；程序持续运行、"
        "自动切换，说明外层 while True 循环生效。"
    )
)

story.append(h2("2.2 完整循环：状态1 → 状态2 → 状态3 → 状态4 → 状态1"))
story.append(
    intro(
        "一次完整循环的输出如下（共 14 秒，即 5 + 2 + 5 + 2）："
    )
)
story.append(
    code_block(
        "交通灯控制系统启动\n"
        "提示：按 1 触发紧急模式，按 0 结束紧急模式\n"
        "当前状态：南北绿灯,东西红灯，持续 5 秒\n"
        "剩余时间：5 秒\n"
        "剩余时间：4 秒\n"
        "剩余时间：3 秒\n"
        "剩余时间：2 秒\n"
        "剩余时间：1 秒\n"
        "切换交通灯...\n"
        "当前状态：南北黄灯,东西红灯，持续 2 秒\n"
        "剩余时间：2 秒\n"
        "剩余时间：1 秒\n"
        "切换交通灯...\n"
        "当前状态：南北红灯,东西绿灯，持续 5 秒\n"
        "剩余时间：5 秒\n"
        "剩余时间：4 秒\n"
        "剩余时间：3 秒\n"
        "剩余时间：2 秒\n"
        "剩余时间：1 秒\n"
        "切换交通灯...\n"
        "当前状态：南北红灯,东西黄灯，持续 2 秒\n"
        "剩余时间：2 秒\n"
        "剩余时间：1 秒\n"
        "切换交通灯...\n"
        "当前状态：南北绿灯,东西红灯，持续 5 秒\n"
        "剩余时间：5 秒\n"
        "剩余时间：4 秒\n"
        "..."
    )
)
story.append(
    body(
        "从输出可以看出，状态依次为状态1 → 状态2 → 状态3 → 状态4 → 状态1，其中绿灯"
        "（状态 1、3）倒计时 5 秒、黄灯（状态 2、4）倒计时 2 秒，最后回到状态 1 重新开始，"
        "验证了状态循环的正确性。"
    )
)

story.append(h2("2.3 触发紧急模式后的运行结果"))
story.append(
    intro(
        "在运行过程中按 1，系统会立即中断当前倒计时并进入紧急模式；按 0 后结束紧急模式，"
        "从状态 1 重新开始。以下是在状态 1 的第 1 秒按下 1 时的运行结果："
    )
)
story.append(
    code_block(
        "交通灯控制系统启动\n"
        "提示：按 1 触发紧急模式，按 0 结束紧急模式\n"
        "当前状态：南北绿灯,东西红灯，持续 5 秒\n"
        "剩余时间：5 秒\n"
        "！！！检测到紧急情况！！！\n"
        "南北：红灯\n"
        "东西：红灯\n"
        "交通灯系统进入紧急模式\n"
        "（按 0 结束紧急模式）\n"
        "紧急模式结束，交通灯系统恢复正常运行\n"
        "切换交通灯...\n"
        "当前状态：南北绿灯,东西红灯，持续 5 秒\n"
        "剩余时间：5 秒\n"
        "剩余时间：4 秒\n"
        "剩余时间：3 秒\n"
        "剩余时间：2 秒\n"
        "剩余时间：1 秒\n"
        "切换交通灯...\n"
        "当前状态：南北黄灯,东西红灯，持续 2 秒\n"
        "剩余时间：2 秒\n"
        "剩余时间：1 秒\n"
        "..."
    )
)
story.append(
    body(
        "可以看到，在状态 1 的倒计时中按下 1 后，程序立即输出“！！！检测到紧急情况！！！”，"
        "南北、东西两个方向都变为红灯并进入紧急模式；按 0 后输出“紧急模式结束”，系统重新从"
        "状态 1（南北绿灯、东西红灯）开始运行。由于程序每 0.05 秒就检测一次按键，无论当前"
        "处于哪个状态，都能很快响应。"
    )
)

# ============ 3 总结 ============
story.append(h1("3 总结"))
story.append(
    Paragraph(
        "本实验用 Python 模拟了十字路口交通灯控制系统：通过“状态 + 持续时间 + 状态切换函数”"
        "实现 4 个状态的自动循环，利用 while 循环让系统持续运行，并使用 msvcrt 非阻塞按键检测"
        "把紧急模式加入正常控制流程。程序实测满足题目要求：正常循环时绿灯 5 秒、黄灯 2 秒；"
        "任意时刻按 1 立即进入全红紧急模式，按 0 后从状态 1 重新开始。",
        summary_style,
    )
)

# ============ 生成 ============
shutil.copyfile(SRC_PDF, BACKUP_PDF)

doc = SimpleDocTemplate(
    OUT_PDF,
    pagesize=A4,
    leftMargin=LEFT,
    rightMargin=RIGHT,
    topMargin=TOP,
    bottomMargin=BOTTOM,
    title="十字路口交通灯控制系统实验报告",
    author="Python_Traffic_Light_Project",
)
doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
print("done")
