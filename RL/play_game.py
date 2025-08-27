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
    
    if keyboard.is_pressed('b'):
        if not last_save_state: 
            with open(save_file, "wb") as f:
                pyboy.save_state(f)
            print(f"✅ 游戏状态已保存至 {save_file}")
            last_save_state = True
    else:
        last_save_state = False

    if (i%200 == 0):
        print(pyboy.memory[0xDBAE],pyboy.memory[0xDBD0])

    if keyboard.is_pressed('q'):
        break

pyboy.stop()
