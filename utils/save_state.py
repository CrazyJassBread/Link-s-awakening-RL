# ...existing code...
from pyboy import PyBoy
from pynput import keyboard
import time

pyboy = PyBoy("game_state/Link's awakening.gb")
load_state = "game_state/Room_51.state"
save_state = "game_state/Room51_task1.state"

try:
    with open(load_state, "rb") as f:
        pyboy.load_state(f)
except FileNotFoundError:
    print("No saved state found")

last_save_state = False
running = True

def on_press(key):
    global last_save_state, running
    try:
        c = getattr(key, "char", None)
        if c:
            c = c.lower()
            if c == "x" and not last_save_state:
                with open(save_state, "wb") as f:
                    pyboy.save_state(f)
                print("Game state saved.")
                last_save_state = True
            if c == "q":
                running = False
                return False  # 停止监听器
    except Exception as e:
        print("键盘处理错误:", e)

def on_release(key):
    global last_save_state
    c = getattr(key, "char", None)
    if c and c.lower() == "x":
        last_save_state = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    for i in range(100000):
        if not running:
            break
        pyboy.tick()
        link = pyboy.get_sprite(2)
        print(f"Link Position: x={link.x}, y={link.y}")
        time.sleep(0.01)
finally:
    listener.stop()
    pyboy.stop()
# ...existing code...