import time
import msvcrt

greenlight_duration = 5    # 绿灯持续秒数
yellowlight_duration = 2   # 黄灯持续秒数

state_text = {
    1: ("南北绿灯,东西红灯"),
    2: ("南北黄灯,东西红灯"),
    3: ("南北红灯,东西绿灯"),
    4: ("南北红灯,东西黄灯"),
}   

def change(state):    #根据当前状态返回下一个状态：1→2→3→4→1
    return state%4 + 1

def duration(state):   #返回指定状态的持续时间：绿灯 5 秒，黄灯 2 秒
    return greenlight_duration if state in (1, 3) else yellowlight_duration

def check_key():
    if msvcrt.kbhit():    #监测是否有字符输入
        return msvcrt.getch()
    return None

def run():
    state = 1
    emergency_mode = False
    print("交通灯控制系统启动")
    print("提示：按 1 触发紧急模式，按 0 结束紧急模式")

    while True:
        # 任务二：检测到 1，立即进入紧急模式
        if emergency_mode:
            print("！！！检测到紧急情况！！！")
            print("南北：红灯")
            print("东西：红灯")
            print("交通灯系统进入紧急模式")
            print("（按 0 结束紧急模式）")

            # 保持全红，直到按下 0
            while True:
                if check_key() == b"0":
                    break
                time.sleep(0.1)    #降低cpu空转

            print("紧急模式结束，交通灯系统恢复正常运行")
            print("切换交通灯...")
            state = 1          # 紧急结束后从状态 1 重新开始
            emergency_mode = False
            continue


        # 正常模式
        text = state_text[state]
        print(f"当前状态：{text}，持续 {duration(state)} 秒")

        for remain in range(duration(state), 0, -1):
            print(f"剩余时间：{remain} 秒")
            for _ in range(20):
                if check_key() == b"1":   # 检测救护车输入
                    emergency_mode = True
                    break
                time.sleep(0.05)
            if emergency_mode:
                break    
            

        if emergency_mode:
            continue            # 立即进入紧急模式，不执行状态切换

        print("切换交通灯...")
        state = change(state)
        
run()


    
