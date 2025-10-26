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

        # 启动仿真与读档
        # 从配置文件（RL/configs/links_awakening.json）或环境变量 LA_TEST 读取 test 标志，
        # 如果为 True 则开启 human 模式（PyBoy window="SDL2"），否则使用无窗口模式("null")
        #self.test = self._load_test_flag()
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
        """
        self.observation_space = spaces.Dict(
            {
                "health": spaces.Box(low=0, high=24, dtype=np.uint8),
                "game_area": spaces.Box(low=0, high=255, shape=(144, 160, 4), dtype=np.uint8),
                "agent_pos": spaces.Box(
                    low=np.array([-200, -200], dtype=np.int16),
                    high=np.array([200, 200], dtype=np.int16),
                    dtype=np.int16,
                ),
            }
        )
        """
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(144, 160, 4), dtype=np.uint8
        )

        # 训练计数
        self.cur_step = 0
        self.episode = 0
        self.disable_render = True

    # ---- Gym API ----
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.cur_step = 0
        self.episode += 1

        with open(self.save_file, "rb") as f:
            self.pyboy.load_state(f)
        self.pyboy.tick(1)

        # 通用状态复位
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

        # 给子类的扩展复位钩子
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

    # ---- 必须由子类实现的钩子 ----
    @abstractmethod
    def check_goal(self) -> bool:
        """任务是否完成"""
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self) -> Tuple[float, bool]:
        """返回 (reward, terminated)"""
        raise NotImplementedError

    # ---- 可由子类覆盖的辅助钩子 ----
    def get_distance(self) -> float:
        """用于同房间内的距离稀疏奖励，默认无距离"""
        return 0.0

    # ---- 通用工具方法（子类可复用） ----
    def preprocess_for_rl(self):
        game_pixels_render = self.pyboy.screen.ndarray[:, :, 0:1]
        game_pixels_render = downscale_local_mean(game_pixels_render, (2, 2, 1)).astype(np.uint8)
        return game_pixels_render

    def _get_obs(self):
        ''' 原本的返回值是dict，现在应对cnn policy改成了box
        return {
            "game_area": np.array(self.pyboy.screen.ndarray(), dtype=np.uint8),
            "health": np.array([self.cur_health], dtype=np.uint8),
            "agent_pos": np.array(self._get_pos(), dtype=np.int16),
        }
        '''
        return self.pyboy.screen.ndarray
    
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
        self.pyboy.tick(8)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(8)

    def is_dead(self) -> bool:
        return self.read_m(ADDR_CUR_HEALTH) == 0

    def is_hurt(self) -> int:
        """返回负的生命变化（受伤为负），否则 0"""
        if not isinstance(self.cur_health, (int, float)) or not isinstance(self.pre_health, (int, float)):
            return 0
        if self.cur_health < self.pre_health:
            return self.cur_health - self.pre_health
        return 0

    def outside_counter_tick(self, max_out: int = 100) -> bool:
        """离开目标房间计数，超过阈值返回 True 并清零"""
        if self.cur_room != self.goal_room:
            self.out_side += 1
        else:
            self.out_side = 0
        if self.out_side >= max_out:
            self.out_side = 0
            return True
        return False

    def tile_explore_bonus(self) -> bool:
        """首次踏入瓦片返回 True"""
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

    def _load_test_flag(self, config_filename: str = "links_awakening.json") -> bool:
        """尝试从环境变量 LA_TEST 或 本仓库下 RL/configs/<config_filename> 的顶层 'test' 字段读取布尔值。
        优先顺序：环境变量 > config 文件中的顶层 'test' 字段。找不到时返回 False。
        """
        # 环境变量优先（方便 CI / 测试覆盖）
        env_val = os.getenv("LA_TEST")
        if env_val is not None:
            val = env_val.strip().lower()
            if val in ("1", "true", "yes", "y", "on"):
                return True
            return False

        # 从项目内的 RL/configs/links_awakening.json 读取（相对于当前文件路径）
        try:
            repo_rl_configs = Path(__file__).resolve().parents[2] / "configs" / config_filename
            if repo_rl_configs.exists():
                with open(repo_rl_configs, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    # 支持顶层 'test' 字段（布尔）
                    if isinstance(cfg, dict) and "test" in cfg:
                        return bool(cfg["test"])
        except Exception:
            # 不要让配置读取阻塞环境初始化；默认为 False
            pass

        return False