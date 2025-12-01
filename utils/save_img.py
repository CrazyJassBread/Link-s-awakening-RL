import io
from pyboy import PyBoy
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard
import torch
import torch.nn as nn
import torch.nn.functional as F
from screen_abstract import gamearea_abstract
import os
from datetime import datetime

pyboy = PyBoy("game_state/Link's awakening.gb")
load_state = "game_state/Room58_task2.state"

try:
    with open(load_state, "rb") as f:
        pyboy.load_state(f)
except FileNotFoundError:
    print("No existing save file, starting new game")

running = True
latest_frame = None  # 新增: 保存当前帧

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _save_frame(array_2d):
    _ensure_dir("img/saved_images")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"img/saved_images/frame_{ts}.png"
    plt.imsave(fname, array_2d, cmap="gray", vmin=0, vmax=255)
    print(f"Saved: {fname}")

def on_press(key):
    global running, latest_frame
    try:
        c = getattr(key, "char", None)
        if c:
            c = c.lower()
            if c == "q":
                running = False
                return False
            elif c == "e":
                if latest_frame is not None:
                    _save_frame(latest_frame)
                else:
                    print("No frame yet.")
    except Exception as e:
        print("键盘处理错误:", e)

listener = keyboard.Listener(on_press=on_press)
listener.start()

"""
plt.ion()
fig, ax = plt.subplots()
image_data = np.zeros((8, 10))
img = ax.imshow(image_data, cmap='gray', vmin=0, vmax=255)

plt.show()
"""

"""
def _get_obs(pyboy):
    screen_tensor = torch.tensor(pyboy.screen.ndarray[:128, :160, 0], dtype=torch.float32)
    pooled = gamearea_abstract(screen_tensor)
    return pooled
"""

def _get_obs(pyboy):
    screen = pyboy.screen.ndarray[:, :, 0]
    return screen

for k in range(100000):
    if not running:
        break
    pyboy.tick()
    sub_screen = _get_obs(pyboy)
    latest_frame = sub_screen  # 新增: 记录当前帧
    #img.set_data(sub_screen)

    #plt.draw()
    #plt.pause(0.001)

listener.stop()
pyboy.stop()