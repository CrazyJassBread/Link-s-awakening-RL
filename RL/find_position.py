import io
import keyboard
from pyboy import PyBoy
import time

def get_link_sprites(game_wrapper):
    """从 game_wrapper 中直接提取林克精灵数据"""
    link_sprites = []
    
    # 通过 botsupport 获取精灵列表（需要 PyBoy 1.5.0+）
    try:
        sprites = game_wrapper.pyboy.botsupport_manager().sprite()
        for sprite in sprites:
            # 林克的特征过滤条件
            if (sprite.shape == (8, 16) and 
               sprite.tiles[0] in [0, 1, 2, 3] and 
               sprite.on_screen):
                link_sprites.append(sprite)
    except AttributeError:
        # 备用方法：直接访问内存中的精灵属性
        for i in range(40):  # Game Boy 最多40个精灵
            x = game_wrapper.pyboy.get_memory_value(0xFE00 + 4*i + 1) - 8
            y = game_wrapper.pyboy.get_memory_value(0xFE00 + 4*i + 0) - 16
            tile = game_wrapper.pyboy.get_memory_value(0xFE00 + 4*i + 2)
            if tile in [0, 1, 2, 3] and 0 <= x < 160 and 0 <= y < 144:
                link_sprites.append({'x': x, 'y': y, 'tile': tile})
    
    return link_sprites

def extract_link_position(game_wrapper):
    """从 game_wrapper 对象解析林克坐标"""
    link_sprites = get_link_sprites(game_wrapper)
    
    # 寻找成对的林克精灵
    for i in range(len(link_sprites)):
        for j in range(i+1, len(link_sprites)):
            s1 = link_sprites[i]
            s2 = link_sprites[j]
            
            # 验证精灵特征
            x_diff = abs(s1.x - s2.x)
            y_diff = abs(s1.y - s2.y)
            
            if x_diff == 8 and y_diff == 0:
                return ((s1.x + s2.x) // 2, (s1.y + s2.y) // 2)
    
    return (None, None)

# 初始化 PyBoy
pyboy = PyBoy("Link's awakening.gb")
save_file = "Link's awakening.gb.state"
with open(save_file, "rb") as f:
    pyboy.load_state(f)  # 使用无头模式
pyboy.game_wrapper.start_game()

try:
    while True:
        # 直接传递 game_wrapper 对象
        link_x, link_y = extract_link_position(pyboy.game_wrapper)
        
        if link_x is not None:
            print(f"\rLink Position: ({link_x}, {link_y})", end="")
            
        pyboy.tick()
        time.sleep(1/60)  # 模拟60FPS

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    pyboy.stop()