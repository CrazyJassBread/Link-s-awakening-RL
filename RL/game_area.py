import io
from pyboy import PyBoy
import numpy as np
import matplotlib.pyplot as plt

pyboy = PyBoy("RL/game_state/Link's awakening.gb")
save_file = "RL/game_state/Room_51.state"

try:
    with open(save_file, "rb") as f:
        pyboy.load_state(f)
except FileNotFoundError:
    print("No existing save file, starting new game")

# 准备matplotlib实时显示
plt.ion()
fig, ax = plt.subplots()
image_data = np.zeros((20, 20))  # 假设你的卷积结果是32x32矩阵
img = ax.imshow(image_data, cmap='gray', vmin=0, vmax=255)
plt.show()

# 阈值划分成三类
thresholds = [85, 170]
np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)

for i in range(4000):
    pyboy.tick()
    game_matrix = pyboy.game_area()
    sub_matrix = game_matrix[:20,:20]
    labels = np.digitize(sub_matrix, thresholds)
    quantized = np.array([0, 128, 255])[labels]
    if i % 100 == 0:
        #print(quantized)
        #print(labels)
        count = np.count_nonzero(labels == 0)
        print((count - 1)//4)
    # print(dtype(game_matrix))
    img.set_data(quantized)
    #img.set_data(sub_matrix)
    plt.draw()
    plt.pause(0.001)
pyboy.stop()