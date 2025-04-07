rom_path = r"Link's awakening.gb"
state = r"Legend_of_Zelda.gb.state"

from pyboy import PyBoy
import os
import numpy as np


def read_game_save(rom_path, save_state_path=None):
    if not os.path.exists(rom_path):
        raise FileNotFoundError(f"ROM文件未找到: {rom_path}")

    with PyBoy(rom_path) as pyboy:
        if save_state_path and os.path.exists(save_state_path):
            with open(save_state_path, "rb") as f:
                pyboy.load_state(f)

        for _ in range(300):
            pyboy.tick()

        np.set_printoptions(
            threshold=np.inf,
            linewidth=200,
            edgeitems=1000
        )

        game_area = pyboy.game_area()
        print(game_area.shape)
        print("完整游戏区域矩阵：")
        print(game_area)
        unique_elements = np.unique(game_area).tolist()
        print(unique_elements)
        for item in unique_elements:
            list_item = [item]
            print(f"tile {item} is used by sprites {pyboy.get_sprite_by_tile_identifier(list_item)}")



read_game_save(rom_path, state)