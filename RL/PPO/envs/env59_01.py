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
        # 怪物统计的初始值
        self.slimes, self.turtles = self._get_monsters()

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.slimes, self.turtles = self._get_monsters()
        return obs, info

    # 任务完成：这里暂以钥匙数量 >= 1 为达成
    def check_goal(self) -> bool:
        return self.read_m(ADDR_KEYS) >= 1

    # 可选：房间内的曼哈顿距离（如用于形状奖励），按你原逻辑自定义
    def get_distance(self) -> float:
        # 示例：对房间 59 自定义一个目标点（需按实际地图修改）
        x, y = self._get_pos()
        target_x, target_y = 30, 45
        return abs(target_x - x) + abs(target_y - y)

    def calculate_reward(self) -> Tuple[float, bool]:
        reward = 0.0
        terminated = False

        # 1) 死亡强惩罚并终止
        if self.is_dead():
            reward -= 1.0
            return reward, True

        # 2) 受伤小惩罚（is_hurt 为负数）
        reward += 0.01 * self.is_hurt()

        # 3) 击杀怪物奖励（基于图像阈值的数量估计）
        reward += self._monster_kill_bonus()

        # 4) 拾取卢比小奖励
        #if self.rupee_gained():
            #reward += 0.5

        # 5) 探索新瓦片微奖励
        #if self.tile_explore_bonus():
            #reward += 0.001

        # 6) 任务完成大奖励并终止
        if self.check_goal():
            reward += 10.0
            terminated = True
        else:
            # 7) 偏离目标房间微惩罚；在目标房间内按距离微惩罚
            if self.cur_room != self.goal_room:
                reward -= 0.0001
            else:
                reward -= 0.0001 * self.get_distance()

        # 8) 长时间离开目标房间：截断式终止并给小惩罚
        if self.outside_counter_tick(max_out=100):
            reward -= 0.01
            terminated = True

        return reward, terminated

    # ---- 房间 51 专属辅助 ----
    def _get_monsters(self):
        # 返回 (slimes, turtles) 的粗略估计
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
        """比对当前与历史数量，减少则给奖励，并更新基准"""
        cur_slimes, cur_turtles = self._get_monsters()
        bonus = 0.0
        if cur_turtles < self.turtles:
            bonus += 5.0
            self.turtles = cur_turtles
        if cur_slimes < self.slimes:
            bonus += 2.0
            self.slimes = cur_slimes
        return bonus