import io
from pyboy import PyBoy
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard

pyboy = PyBoy("game_state/Link's awakening.gb")
load_state = "game_state/Room51_task1.state"

try:
    with open(load_state, "rb") as f:
        pyboy.load_state(f)
except FileNotFoundError:
    print("No existing save file, starting new game")

running = True

def on_press(key):
    global last_save_state, running
    try:
        c = getattr(key, "char", None)
        if c:
            c = c.lower()
            if c == "q":
                running = False
                return False  # 停止监听器
    except Exception as e:
        print("键盘处理错误:", e)


listener = keyboard.Listener(on_press=on_press)
listener.start()

# 准备matplotlib实时显示
plt.ion()
fig, ax = plt.subplots()
image_data = np.zeros((20, 20)) 
img = ax.imshow(image_data, cmap='gray', vmin=0, vmax=255)
plt.show()

# 阈值划分成三类
thresholds = [85, 170]
np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)

for i in range(10000):
    if not running:
        break
    pyboy.tick()
    game_matrix = pyboy.game_area()
    sub_matrix = game_matrix[:20,:20]
    labels = np.digitize(sub_matrix, thresholds)
    quantized = np.array([0, 128, 255])[labels]
    if i % 100 == 0:
        count = np.count_nonzero(labels == 0)
        # print((count - 1)//4)
    img.set_data(sub_matrix)
    plt.draw()
    plt.pause(0.001)

listener.stop()
pyboy.stop()
