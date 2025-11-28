from __future__ import annotations
from typing import Tuple
from .base_env import BaseEnv, ADDR_KEYS

class Room51_Task1_Env(BaseEnv):
    """
    51 Task one:
        踩下按钮（80，45）
        打开宝箱（130，40）
        获得key
    """
    def __init__(self, game_file: str, save_file: str, render_mode: str | None = None, goal_room: int | None = 51):
        super().__init__(game_file, save_file, goal_room=goal_room, render_mode=render_mode)
        # 怪物统计的初始值
        self.flag = False
        self.pre_distance1 = self.get_distance(80, 45)
        self.pre_distance2 = self.get_distance(130, 40)

    def _reset_extra(self, options=None):
        self.flag = False

    # 任务完成：这里暂以钥匙数量 >= 1 为达成
    def check_goal(self) -> bool:
        return self.read_m(ADDR_KEYS) >= 1

    # 可选：房间内的曼哈顿距离（如用于形状奖励），按你原逻辑自定义
    def get_distance(self, target_x: int | None = None, target_y: int | None = None) -> float:
        # 示例：对房间 51 自定义一个目标点（需按实际地图修改）
        x, y = self._get_pos()
        if target_x is not None and target_y is not None:
            return abs(target_x - x) + abs(target_y - y)
        
        if self.flag == False:
            target_x, target_y = 80, 45
        else:
            target_x, target_y = 130,40
        return abs(target_x - x) + abs(target_y - y)

    def update_flag(self) -> bool:
        x, y = self._get_pos()
        # 检测按钮是否踩下
        if self.flag == False and abs(x - 80) < 3 and abs(y - 45) < 3:
            self.flag = True
        return self.flag

    def calculate_reward(self) -> Tuple[float, bool]:
        reward = 0.0
        terminated = False

        # 1) 死亡强惩罚并终止
        if self.is_dead():
            reward -= 1.0
            return reward, True
        

        # 2) 受伤小惩罚（is_hurt 为负数）
        reward += 0.01 * self.is_hurt()
        
        # 3) 踩下按钮奖励
        if self.flag == False and self.update_flag():
            reward += 5.0

        # 4) 探索新瓦片微奖励
        if self.tile_explore_bonus():
            reward += 0.01

        # 5) 任务完成大奖励并终止
        if self.check_goal():
            reward += 10.0
            terminated = True

        if self.flag == False:
            reward -= (self.get_distance() / self.pre_distance1)* 0.01
        else:
            reward -= (self.get_distance() / self.pre_distance2) * 0.01

        # 8) 长时间离开目标房间：截断式终止并给小惩罚
        if self.outside_counter_tick(max_out=100):
            reward -= 0.1
            terminated = True

        return reward, terminated