import io
import keyboard
from pyboy import PyBoy
import numpy as np
pyboy = PyBoy("RL\game_state\Link's awakening.gb")
save_file = "RL\game_state\Link's awakening.gb.state"

try:
    with open(save_file, "rb") as f:
        pyboy.load_state(f)
except FileNotFoundError:
    print("No existing save file, starting new game")

last_save_state = False

for i in range(10000):
    pyboy.tick()
    
    if keyboard.is_pressed('x'):
        if not last_save_state: 
            with open(save_file, "wb") as f:
                pyboy.save_state(f)
            print(f"✅ 游戏状态已保存至 {save_file}")
            last_save_state = True
    else:
        last_save_state = False

    if (i%200 == 0):
        #print(pyboy.memory[0xDB5A])
        
        print("###############################")
        frame = pyboy.game_area()   # 或者 screen_image().convert('RGB') -> np.array
        #print(type(frame))       # 查看数据类型（应该是 numpy.ndarray）
        print(frame.shape)       # 数组形状，例如 (144, 160, 3)
        #print(frame.dtype)
        np.set_printoptions(threshold=np.inf)  # 关闭省略，打印完整数组
        print(frame)
        #print(pyboy.game_area())
        #print(sprite)
        #print (pyboy.memory[0xDBAE])
    if keyboard.is_pressed('q'):
        break

pyboy.stop()
