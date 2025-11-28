import io
from pyboy import PyBoy
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard
import torch
import torch.nn as nn
import torch.nn.functional as F
from screen_abstract import gamearea_abstract


pyboy = PyBoy("game_state/Link's awakening.gb")
load_state = "game_state/Room58_task2.state"

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
image_data = np.zeros((8, 10)) 
img = ax.imshow(image_data, cmap='gray', vmin=0, vmax=255)

# 插入数值标注
text_annotations = []
for i in range(image_data.shape[0]):
    row_annotations = []
    for j in range(image_data.shape[1]):
        text = ax.text(j, i, f"{image_data[i, j]:.1f}", ha="center", va="center", color="red", fontsize=8)
        row_annotations.append(text)
    text_annotations.append(row_annotations)

plt.show()

def _get_obs(pyboy):
    screen_tensor = torch.tensor(pyboy.screen.ndarray[:128, :160, 0], dtype=torch.float32)
    pooled = gamearea_abstract(screen_tensor)
    return pooled

for k in range(100000):
    if not running:
        break
    pyboy.tick()
    sub_screen = _get_obs(pyboy)
    # game_matrix = pyboy.game_area()
    # sub_matrix = game_matrix[:20,:20]
    img.set_data(sub_screen)

    for i in range(sub_screen.shape[0]):
        for j in range(sub_screen.shape[1]):
            text_annotations[i][j].set_text(f"{sub_screen[i, j]:.1f}")

    plt.draw()
    plt.pause(0.001)

listener.stop()
pyboy.stop()
