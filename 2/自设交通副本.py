import time

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

def run():
    state = 1
    print("交通灯控制系统启动")

    while True:
        text = state_text[state]
        print(f"当前状态：{text}，持续 {duration(state)} 秒")
        
        for remain in range(duration(state), 0, -1):
                print(f"剩余时间：{remain} 秒", end="\r")
                time.sleep(1)
        
        print("切换交通灯...")
        state = change(state)
        
run()


    