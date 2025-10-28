from __future__ import annotations
from typing import Tuple
import numpy as np
from .base_env import BaseEnv, ADDR_KEYS

class Room59_Task1_Env(BaseEnv):

    """
    59 Task one:（对应state文件 Room59_task1.state）
        杀死两只乌龟怪物
        走到位置(30,45)处
        获得key
    """

    def __init__(self, game_file: str, save_file: str):
        super().__init__(game_file, save_file, goal_room=59)
        self.slimes, self.turtles = self._get_monsters()

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.slimes, self.turtles = self._get_monsters()
        return obs, info

    def check_goal(self) -> bool:
        return self.read_m(ADDR_KEYS) >= 1

    def get_distance(self) -> float:
        x, y = self._get_pos()
        target_x, target_y = 30, 45
        return abs(target_x - x) + abs(target_y - y)

    def calculate_reward(self) -> Tuple[float, bool]:
        reward = 0.0
        terminated = False

        if self.is_dead():
            reward -= 1.0
            return reward, True

        reward += 0.01 * self.is_hurt()

        reward += self._monster_kill_bonus()

        if self.check_goal():
            reward += 100.0
            terminated = True
        else:
            if self.cur_room != self.goal_room:
                reward -= 0.001
            else:
                reward += 0.001 * ((200 - self.get_distance()) / 200)

        if self.outside_counter_tick(max_out=100):
            reward -= 0.1
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