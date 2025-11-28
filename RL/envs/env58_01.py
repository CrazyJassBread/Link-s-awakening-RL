from __future__ import annotations
from typing import Tuple
import numpy as np
from .base_env import BaseEnv, ADDR_KEYS

class Room58_Task1_Env(BaseEnv):
    """
    58 Task one:（对应state文件 Room58_task1.state）
        走到位置(30,45)处
        获得key
    """
    def __init__(self, game_file: str, save_file: str, render_mode: str | None = None, goal_room: int | None = 58):
        super().__init__(game_file, save_file, goal_room=goal_room, render_mode=render_mode)
        # self.slimes, self.turtles = self._get_monsters()
        self.target_pos = (30,45)
        self.pre_distance = None
        self.cur_distance = None
        

    def _reset_extra(self, seed=None, options=None):
        self.pre_distance = self.get_distance()
        self.cur_distance = self.pre_distance

    def check_goal(self) -> bool:
        return self.read_m(ADDR_KEYS) >= 1

    def get_distance(self) -> float:
        x, y = self._get_pos()
        target_x, target_y = self.target_pos
        return abs(target_x - x) + abs(target_y - y)
    
    def _step_extra(self):
        self.cur_distance = self.get_distance()

    def calculate_reward(self) -> Tuple[float, bool]:
        reward = 0.0
        terminated = False

        if self.is_dead():
            reward -= 1.0
            return reward, True

        reward += 0.01 * self.is_hurt()
        if self.cur_distance < self.pre_distance and self.cur_distance != None and self.cur_room == 58:
            reward += 0.05

        if self.check_goal():
            reward += 10.0
            terminated = True
        else:
            if self.cur_room != self.goal_room:
                reward -= 0.05

        if self.outside_counter_tick(max_out = 200):
            reward -= 1.0
            terminated = True

        return reward, terminated
