# -*- coding: utf-8 -*-
"""
十字路口交通灯控制系统

任务一：正常模式                                                                 
    系统包含 4 个状态，按 1 → 2 → 3 → 4 → 1 不断循环：
        状态1：南北绿灯，东西红灯（绿灯持续 5 秒）
        状态2：南北黄灯，东西红灯（黄灯持续 2 秒）
        状态3：南北红灯，东西绿灯（绿灯持续 5 秒）
        状态4：南北红灯，东西黄灯（黄灯持续 2 秒）

任务二：紧急模式
    运行过程中输入 1：检测到救护车，无论当前处于什么状态，立即进入
    南北红灯、东西红灯的紧急模式；输入 0：结束紧急模式，系统重新从
    状态1（南北绿灯，东西红灯）开始运行。
"""

import time
from threading import Thread

GREEN_DURATION = 5    # 绿灯持续秒数
YELLOW_DURATION = 2   # 黄灯持续秒数

# 状态编号 → (南北方向灯色, 东西方向灯色)
STATE_TEXT = {
    1: ("南北绿灯", "东西红灯"),
    2: ("南北黄灯", "东西红灯"),
    3: ("南北红灯", "东西绿灯"),
    4: ("南北红灯", "东西黄灯"),
}

# 全局紧急标志：后台输入线程修改，主循环读取
emergency = False


def change_state(state):
    """根据当前状态返回下一个状态：1→2→3→4→1"""
    return state % 4 + 1


def state_duration(state):
    """返回指定状态的持续时间：绿灯 5 秒，黄灯 2 秒"""
    return GREEN_DURATION if state in (1, 3) else YELLOW_DURATION


def input_listener():
    """后台线程：监听键盘输入，控制紧急模式的进入与结束"""
    global emergency
    while True:
        try:
            command = input().strip()
        except EOFError:   # 输入流关闭时安全退出
            return
        if command == "1":
            emergency = True
        elif command == "0":
            emergency = False


def run():
    """交通灯主控制循环"""
    global emergency

    # 启动键盘监听线程
    Thread(target=input_listener, daemon=True).start()

    state = 1
    print("===== 十字路口交通灯系统 =====")
    print("提示：输入 1 触发紧急模式，输入 0 结束紧急模式\n")

    while True:
        # ---------- 紧急模式 ----------
        if emergency:
            print("\n！！！检测到紧急情况！！！")
            print("南北：红灯")
            print("东西：红灯")
            print("交通灯系统进入紧急模式")
            print("（输入 0 结束紧急模式）\n")

            # 保持全红，等待输入 0
            while emergency:
                time.sleep(1)

            # 紧急状态结束，重新从状态 1 开始
            print("\n紧急模式结束，交通灯系统恢复正常运行")
            print("切换交通灯...\n")
            state = 1
            continue

        # ---------- 正常模式 ----------
        ns_text, ew_text = STATE_TEXT[state]
        print(f"当前状态：{ns_text} | {ew_text}")

        for remaining in range(state_duration(state), 0, -1):
            print(f"剩余时间：{remaining}秒")
            time.sleep(1)          # 模拟真实世界中经过 1 秒
            if emergency:          # 每秒检查一次是否有紧急情况
                break

        if emergency:
            continue               # 立即进入紧急模式，不切换状态

        print("切换交通灯...\n")
        state = change_state(state)  # 切换下一个状态


if __name__ == "__main__":
    run()
