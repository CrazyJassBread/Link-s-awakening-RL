from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from pyboy import PyBoy
import json
import os
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
from pyboy.utils import WindowEvent
from skimage.transform import downscale_local_mean
from .screen_abstract import gamearea_abstract
import torch
# 通用常量
TOTAL_STEPS = 3_000_000
MAX_STEPS = 5_000

# 内存地址（命名化，便于复用/维护）
ADDR_CUR_HEALTH = 0xDB5A
ADDR_MAX_HEALTH = 0xDB5B
ADDR_RUPEE      = 0xDB5E
ADDR_ROOM_ID    = 0xDBAE
ADDR_KEYS       = 0xDBD0

class BaseEnv(gym.Env, ABC):
    """
    Zelda 抽象环境基类：封装通用流程，子类则实现具体任务逻辑
    为什么这样封装🤔
    因为原来的代码快成尸山了
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, game_file: str, save_file: str, goal_room: int | None = None):
        super().__init__()
        self.game_file = game_file
        self.save_file = save_file

        # XXX：新增test来表示是测试还是训练
        self.test = False
        window_mode = "SDL2" if self.test else "null"
        self.pyboy = PyBoy(game_file, sound_emulated=False, window=window_mode)
        try:
            with open(save_file, "rb") as f:
                self.pyboy.load_state(f)
        except FileNotFoundError:
            print("No existing save file, starting new game")

        # 状态变量
        self.max_health = self.read_m(ADDR_MAX_HEALTH)
        self.pre_health = self.read_m(ADDR_CUR_HEALTH)
        self.cur_health = self.pre_health

        self.pre_rupee = self.read_m(ADDR_RUPEE)
        self.cur_rupee = self.pre_rupee

        self.cur_room = self.read_m(ADDR_ROOM_ID)
        self.goal_room = self.cur_room if goal_room is None else goal_room
        self.out_side = 0

        self.visited_rooms: set[int] = set()
        self.visited_tiles: set[tuple[int, int, int]] = set()  # (room_id, tile_x, tile_y)
        
        # 计算距离变化信息
        self.target_pos = None
        self.pre_distance = None
        self.cur_distance = None
        
        # 动作与观察空间
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]
        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(8, 10), dtype=np.uint8  # 修改为卷积后的形状
        )
        # 训练计数
        self.cur_step = 0
        self.episode = 0
        self.disable_render = True

    # Gym API 接口
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.cur_step = 0
        self.episode += 1

        with open(self.save_file, "rb") as f:
            self.pyboy.load_state(f)
        self.pyboy.tick(1)

        # 通用状态
        self.cur_room = self.read_m(ADDR_ROOM_ID)
        if self.goal_room is None:
            self.goal_room = self.cur_room
        self.visited_rooms.clear()
        self.visited_tiles.clear()

        self.pre_health = self.read_m(ADDR_CUR_HEALTH)
        self.cur_health = self.pre_health
        self.pre_rupee = self.read_m(ADDR_RUPEE)
        self.cur_rupee = self.pre_rupee
        self.out_side = 0

        # 给子类的扩展复位选择
        self._reset_extra(options)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    # 子类覆盖此钩子做额外复位（默认无操作）
    def _reset_extra(self, options: dict | None):
        pass

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action!"
        self.cur_step += 1

        self.pre_health = self.cur_health
        self.pre_rupee = self.cur_rupee

        self.run_action(action)
        self._step_extra()
        # 更新核心状态
        self.cur_health = self.read_m(ADDR_CUR_HEALTH)
        self.cur_room = self.read_m(ADDR_ROOM_ID)
        self.cur_rupee = self.read_m(ADDR_RUPEE)
        self.visited_rooms.add(self.cur_room)

        # 奖励与终止判定（由子类决定奖励构成）
        reward, terminated = self.calculate_reward()

        # 截断：通用的最大步数限制
        truncated = self.cur_step >= MAX_STEPS

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info
    
    def _step_extra(self):
        pass

    def render(self, mode: str = "human"):
        if mode == "human" and not getattr(self, "disable_render", False):
            self.pyboy.screen.ndarray
        elif mode == "rgb_array":
            frame = self.pyboy.screen.ndarray
            return frame
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        self.pyboy.stop()

    @abstractmethod
    def check_goal(self) -> bool:
        """任务是否完成"""
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self) -> Tuple[float, bool]:
        """返回 (reward, terminated)"""
        raise NotImplementedError

    # ---- 可由子类覆盖 ----
    def get_distance(self) -> float:
        """默认无距离"""
        return 0.0

    def _get_obs(self):
        # 当前的方案是对 screen 进行一个 16 * 16 的高斯卷积操作，降维到（8 * 10）
        screen_tensor = torch.tensor(self.pyboy.screen.ndarray[:128, :160, 0], dtype=torch.float32)
        pooled = gamearea_abstract(screen_tensor)
        return pooled
    
    def _get_info(self):
        return {
            "goal": bool(self.check_goal()),
            "room": int(self.cur_room),
        }

    def _get_pos(self) -> Tuple[int, int]:
        sprite = self.pyboy.get_sprite(2)
        return sprite.x, sprite.y

    def _get_tile(self) -> Tuple[int, int]:
        x, y = self._get_pos()
        tile_x = int(max(0, x) // 8)
        tile_y = int(max(0, y) // 8)
        return tile_x, tile_y

    def run_action(self, action: int):
        self.pyboy.send_input(self.valid_actions[action])
        self.pyboy.tick(10)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(10)

    def is_dead(self) -> bool:
        return self.read_m(ADDR_CUR_HEALTH) == 0

    def is_hurt(self) -> int:
        """返回生命变化，非负值化为0"""
        if not isinstance(self.cur_health, (int, float)) or not isinstance(self.pre_health, (int, float)):
            return 0
        if self.cur_health < self.pre_health:
            return self.cur_health - self.pre_health
        return 0

    def outside_counter_tick(self, max_out: int = 100) -> bool:
        """离开目标房间计数，超过阈值返回 True """
        if self.cur_room != self.goal_room:
            self.out_side += 1
        else:
            self.out_side = 0
        if self.out_side >= max_out:
            self.out_side = 0
            return True
        return False

    def tile_explore_bonus(self) -> bool:
        """探索区域奖励"""
        if self.cur_room == self.goal_room:
            tile_x, tile_y = self._get_tile()
            key = (int(self.cur_room), tile_x, tile_y)
            if key not in self.visited_tiles:
                self.visited_tiles.add(key)
                return True
        return False

    def rupee_gained(self) -> bool:
        """检测卢比是否增长"""
        gained = self.cur_rupee > self.pre_rupee
        if gained:
            self.pre_rupee = self.cur_rupee
        return gained

    def read_m(self, addr: int) -> int:
        return self.pyboy.memory[addr]