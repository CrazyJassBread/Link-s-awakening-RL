import io
import keyboard
from pyboy import PyBoy

pyboy = PyBoy("Link's awakening.gb")
save_file = "Link's awakening.gb.state"

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

    if (i%100 == 0):
        #print(pyboy.memory[0xDB5A])
        sprite = pyboy.get_sprite(2)
        x = sprite.x
        y = sprite.y
        print(f"人物位置：({x}, {y})")
        #print(sprite)

    if keyboard.is_pressed('q'):
        break

pyboy.stop()
