import io
import keyboard
from pyboy import PyBoy
import numpy as np

from skimage.transform import downscale_local_mean

pyboy = PyBoy("RL\game_state\Link's awakening.gb")
save_file = "RL\game_state\Link's awakening.gb.state"

try:
    with open(save_file, "rb") as f:
        pyboy.load_state(f)
except FileNotFoundError:
    print("No existing save file, starting new game")

last_save_state = False

def render(image):
    game_pixels_render = image[:,:,0:1]  # (144, 160, 3)

    game_pixels_render = (
        downscale_local_mean(game_pixels_render, (2,2,1))
    ).astype(np.uint8)
    return game_pixels_render

for i in range(100000):
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
        #print(pyboy.memory[0xDBAE],pyboy.memory[0xDBD0])
        #frame = render(pyboy.screen.ndarray)
        #frame = pyboy.screen.ndarray
        #print(frame.shape)
        #print(frame.dtype)
        #print(type(frame))
        #print(type(photo))
        #print(frame)
        #print(pyboy.get_sprite(2).x,pyboy.get_sprite(2).y)
        print(pyboy.memory[0xDBAE])

    if keyboard.is_pressed('q'):
        break

pyboy.stop()
