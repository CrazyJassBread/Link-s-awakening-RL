import numpy as np
from pyboy import PyBoy
from pynput import keyboard
import time

pyboy = PyBoy("game_state/Link's awakening.gb")
load_state = "game_state/Room58_task2.state"

try:
    with open(load_state, "rb") as f:
        pyboy.load_state(f)
except FileNotFoundError:
    print("No saved state found")

running = True

def on_press(key):
    global running
    try:
        c = getattr(key, "char", None)
        if c:
            c = c.lower()
            if c == "q":
                running = False
                return False  # 停止监听器
    except Exception as e:
        print("键盘处理错误:", e)

def on_release(key):
    pass
    
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def _get_monsters(pyboy):
    game_area = pyboy.game_area()
    sub_area = game_area[:20, :20]
    thresholds = [85, 170]
    labels = np.digitize(sub_area, thresholds)
    count_0 = np.count_nonzero(labels == 0)
    turtles = (count_0 - 1) // 4
    return turtles

try:
    for i in range(100000):
        if not running:
            break
        pyboy.tick()
       
        if i % 100 == 0:
            monsters = _get_monsters(pyboy)
            link = pyboy.get_sprite(2)
            print(f"Monsters remaining: {monsters}")
            print(f"Link Position: x={link.x}, y={link.y}")
            print(f"current room: {pyboy.memory[0xDBAE]}")
            print (f"current health: {pyboy.memory[0xDB5A]}")
        time.sleep(0.01)
finally:
    listener.stop()
    pyboy.stop()
