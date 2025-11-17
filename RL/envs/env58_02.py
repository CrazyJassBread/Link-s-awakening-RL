from __future__ import annotations
from typing import Tuple
import numpy as np
from .base_env import BaseEnv, ADDR_KEYS

class Room58_Task2_Env(BaseEnv):
    """
    58 Task two:（对应state文件 Room58_task2.state）
        杀死两只乌龟怪物
        走到位置(30,45)处
        获得key
    """
    def __init__(self, game_file: str, save_file: str):
        super().__init__(game_file, save_file, goal_room=58)
        self.slimes, self.turtles = self._get_monsters()
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

        reward += self._monster_kill_bonus()
        
        if self.cur_distance < self.pre_distance and self.cur_distance != None and self.cur_room == 58:
            reward += 0.5

        if self.check_goal():
            reward += 100.0
            terminated = True
        else:
            if self.cur_room != self.goal_room:
                reward -= 0.1

        if self.outside_counter_tick(max_out=100):
            reward -= 5
            terminated = True

        return reward, terminated

    # 房间 59 击杀怪物辅助
    def _get_monsters(self):
        game_area = self.pyboy.game_area()
        sub_area = game_area[:20, :20]
        thresholds = [85, 170]
        labels = np.digitize(sub_area, thresholds)
        count_0 = np.count_nonzero(labels == 0)
        count_1 = np.count_nonzero(labels == 1)
        slimes = (count_1 + 1) // 2
        turtles = (count_0 - 1) // 4
        return slimes, turtles

    def _monster_kill_bonus(self) -> float:
        cur_slimes, cur_turtles = self._get_monsters()
        bonus = 0.0
        if cur_turtles < self.turtles:
            bonus += 10.0
            self.turtles = cur_turtles
        if cur_slimes < self.slimes:
            bonus += 4.0
            self.slimes = cur_slimes
        return bonus