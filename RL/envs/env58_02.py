from __future__ import annotations
from typing import Tuple
import numpy as np
from .base_env import BaseEnv

class Room58_Task2_Env(BaseEnv):
    def __init__(self, game_file: str, save_file: str):
        super().__init__(game_file, save_file, goal_room=58)
        self.turtles = self._get_monsters()

    def check_goal(self) -> bool:
        _, y = self._get_pos()
        if self.cur_room != 58 or y == -16:
            return False
        else:
            return self.turtles == 0

    def calculate_reward(self) -> Tuple[float, bool]:
        reward = 0.0
        terminated = False
        _, y = self._get_pos()
        if self.is_dead():
            reward -= 1.0
            return reward, True

        reward += 0.01 * self.is_hurt()

        if y != -16:
            reward += self._monster_kill_bonus()

        if self.check_goal():
            reward += 100.0
            terminated = True
        else:
            if self.cur_room != self.goal_room:
                reward -= 0.1

        if self.outside_counter_tick(max_out=100):
            reward -= 1
            terminated = True

        return reward, terminated

    def _get_monsters(self):
        game_area = self.pyboy.game_area()
        sub_area = game_area[:20, :20]
        thresholds = [85, 170]
        labels = np.digitize(sub_area, thresholds)
        count_0 = np.count_nonzero(labels == 0)
        turtles = (count_0 - 1) // 4
        return turtles

    def _monster_kill_bonus(self) -> float:
        cur_turtles = self._get_monsters()
        bonus = 0.0
        if cur_turtles < self.turtles:
            bonus += 10.0
        self.turtles = cur_turtles
        return bonus