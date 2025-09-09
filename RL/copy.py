import numpy as np
import random
from pyboy import PyBoy
import gymnasium as gym
from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from skimage.transform import downscale_local_mean

save_file = "Zelda-Link-s-awakening-agent_copy/RL/game_state/Room_51.state"
game_file = "Zelda-Link-s-awakening-agent_copy/RL/game_state/Link's awakening.gb"


TOTAL_STEPS = 1000000
MAX_STEPS = 5000


class Zelda_Env(gym.Env):
    def __init__(self, game_file, save_file, task_name=None, task_params=None, terminate_on_subtask=False):

        """初始化游戏环境"""
        super().__init__()
        self.pyboy = PyBoy(game_file, sound_emulated = False)
        try:
            with open(save_file, "rb") as f:
                self.pyboy.load_state(f)
        except FileNotFoundError:
            print("No existing save file, starting new game")
        
        """游戏状态参数设置"""
        # 记录agent行动前的血量和行动后的血量
        self.max_health = self.read_m(0xDB5B)
        self.pre_health = self.read_m(0xDB5A)
        self.cur_health = self.pre_health

        # XXX 似乎可以通过持有的卢比数目来判断是否击杀怪物
        self.pre_rupee = self.read_m(0xDB5E)
        self.cur_rupee = self.read_m(0xDB5E)

       # 钥匙数量（背包内钥匙计数）
        self.pre_keys = self.read_m(0xDBD0)
        self.cur_keys = self.pre_keys

        # 当前所处的迷宫房间号
        self.goal_room = self.read_m(0xDBAE)
        self.cur_room = self.read_m(0xDBAE)
        self.out_side = 0 # XXX 计算agent脱离目标房间的时间
        # XXX 新增房间目标完成检测
        self.cur_goal = False
        self.visited_rooms = set()
         # 记录每回合在每个房间中访问过的网格（tile），用于探索奖励
        self.visited_tiles = set()  # 元素形式：(room_id, tile_x, tile_y)
        self.explore_bonus = 0.02  # 探索新网格的正向奖励规模
       
        # 记录每回合在每个房间中访问过的网格（tile），用于探索奖励
        self.visited_tiles = set()  # 元素形式：(room_id, tile_x, tile_y)
        self.explore_bonus = 0.002  # 探索新网格的正向奖励规模（默认，可被任务权重覆盖）
        # 按钮检测辅助：在区域内连续停留若干步视为按下
        self.button_dwell = 0
        self.button_pressed = False
        # 子任务设置
        self.task_name = task_name  # 可选: 'get_key', 'reach_area', 'press_button', 'kill_enemy', 'explore_tiles'
        self.task_params = task_params or {}
        self.terminate_on_subtask = terminate_on_subtask
        self.task_dwell_steps = 0  # press_button 等任务用

 # 任务奖励权重配置（可被 task_params['reward_weights'] 覆盖）
        self.reward_profiles = {
            "default": {
                "death_penalty": 1.0,
                "hurt_coef": 0.01,
                "rupee_reward": 1.0,
                "goal_reward": 10.0,
                "room_penalty": 0.001,
                "distance_coef": 0.0001,
                "explore_bonus": 0.002,
                "get_key_reward": 20.0,
                "press_button_reward": 3.0,
                "outside_penalty": 0.1,
            },
            # 强调拿钥匙与探索
            "get_key": {
                "death_penalty": 1.0,
                "hurt_coef": 0.01,
                "rupee_reward": 0.5,
                "goal_reward": 0.0,
                "room_penalty": 0.0005,
                "distance_coef": 0.00005,
                "explore_bonus": 0.003,
                "outside_penalty": 0.1,
            },
             # 用卢比上涨近似击杀信号
            "kill_enemy": {
                "death_penalty": 1.0,
                "hurt_coef": 0.01,
                "rupee_reward": 2.0,
                "goal_reward": 0.0,
                "room_penalty": 0.0005,
                "distance_coef": 0.0,
                "explore_bonus": 0.001,
                "outside_penalty": 0.1,
            },
            # 主要依靠子任务本身的区域塑形
            "press_button": {
                "death_penalty": 1.0,
                "hurt_coef": 0.01,
                "rupee_reward": 0.0,
                "goal_reward": 0.0,
                "room_penalty": 0.0005,
                "distance_coef": 0.0,
                "explore_bonus": 0.001,
                "outside_penalty": 0.1,
            },
             # 主要依靠 reach_area 子任务塑形
            "reach_area": {
                "death_penalty": 1.0,
                "hurt_coef": 0.01,
                "rupee_reward": 0.0,
                "goal_reward": 0.0,
                "room_penalty": 0.1,
                "distance_coef": 0.0,
                "explore_bonus": 0.001,
                "outside_penalty": 0.1,
            },
            # 加强探索
            "explore_tiles": {
                "death_penalty": 1.0,
                "hurt_coef": 0.01,
                "rupee_reward": 0.0,
                "goal_reward": 0.0,
                "room_penalty": 0.01,
                "distance_coef": 0.0,
                "explore_bonus": 0.004,
                "outside_penalty": 0.1,
            },
        }
        #self.zelda = self.pyboy.game_wrapper
        # 设置不同房间的任务目标
        self.room_goals = {
            59: "leave current room", # 59号房间是迷宫入口房间
            58: "kill enemy and get key", # 迷宫入口左侧房间，有两个乌龟怪物
            51: "kill turtle,push button and open box" # 有一个凹型陷阱，需要绕过陷阱并且击败怪物
        }
         # 房间内按钮区域近似（像素坐标）: room_id -> [x1, y1, x2, y2], dwell_steps
        self.button_area_map = {
            51: {"area": [28, 36, 40, 48], "dwell": 20},
            58: {"area": [30, 40, 44, 56], "dwell": 15},
        }


        """动作空间设定"""
        # 定义动作空间和观察空间
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            #WindowEvent.PRESS_BUTTON_START
        ]
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            #WindowEvent.RELEASE_BUTTON_START
        ]
        self.action_space = spaces.Discrete(len(self.valid_actions))

        # obs空间设置
        self.observation_space = spaces.Dict(
            {
                "health": spaces.Box(low = 0,high = 24,dtype = np.uint8),
                "screen": spaces.Box(low = 0,high = 255, shape = (72,80,1),dtype = np.uint8),
                "agent_pos" : spaces.Box(
                    low=np.array([-200, -200], dtype=np.int16),        # x 最小 0, y 最小 0
                    high=np.array([200, 200], dtype=np.int16),
                    dtype = np.int16)
            }
        )
        """训练参数设置""" 
        self.reward = 0
        self.cur_step = 0
        self.episode = 0

    def reset(self, seed = None, options = None):
        # SB3必须传入seed参数，否则会报错
        super().reset(seed = seed)
        if seed is not None:
            np.random.seed(seed)
        self.cur_step = 0
        self.episode += 1
        self.reward = 0

        #这里采用更方便的方式，及直接使用stateload来重置游戏
        #self.pyboy.send_input(WindowEvent.STATE_LOAD) 
        with open(save_file, "rb") as f:
            self.pyboy.load_state(f)
        self.pyboy.tick(1)
        #self.pyboy.send_input(WindowEvent.STATE_LOAD)
        
        # 重置其他状态参数
        # 重置走过的房间编号
        self.goal_room = self.read_m(0xDBAE)
        self.cur_room = self.goal_room
        self.visited_rooms = set()
        self.visited_tiles = set()
        self.button_dwell = 0
        self.button_pressed = False
        self.task_dwell_steps = 0
        # 允许通过 reset(options) 在每回合切换任务
        if options is not None:
            if 'task_name' in options:
                self.task_name = options.get('task_name')
            if 'task_params' in options:
                self.task_params = options.get('task_params') or {}
            if 'terminate_on_subtask' in options:
                self.terminate_on_subtask = bool(options.get('terminate_on_subtask'))


        self.pre_health = self.read_m(0xDB5A)
        self.cur_health = self.pre_health
         # 重置键值计数
        self.pre_keys = self.read_m(0xDBD0)
        self.cur_keys = self.pre_keys


        #self.goal_room = self.read_m(0xDBAE)
        #self.cur_room = self.read_m(0xDBAE)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    # 游戏画面渲染设置
    def render(self, mode = "human"):
        if mode == "human":
            self.pyboy.render_screen()
        elif mode == "rgb_array":
            frame = self.pyboy.screen.ndarray
            # 返回 (H, W, 3) 的 uint8 图像
            if frame.ndim == 2:
                frame = np.stack([frame, frame, frame], axis=-1)
            elif frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            return frame.astype(np.uint8)
    
    
    # 处理游戏screen，降低维度并且缩放至size大小，以备后续rl训练使用
    def preprocess_for_rl(self):
        game_pixels_render = self.pyboy.screen.ndarray[:,:,0:1]
        game_pixels_render = (
            downscale_local_mean(game_pixels_render,(2,2,1))
        ).astype(np.uint8)
        return game_pixels_render

    """
    def _get_obs(self):
        读取游戏当前状态，并将其转换成易于读取的格式
        # 目前训练任务下 obs不需要返回过多的信息
        cur_screen = self.preprocess_for_rl()
        observation = {
            "screen": cur_screen,
            "health": self.cur_health,
            "agent_pos": self._get_pos()
            #"key" : self.read_m(0xDBD0), # 读取钥匙数量，1表示成功拿到钥匙
            #"room" :self.read_m(0xDBAE), # 当前所处的房间位置
            #"ItemA": self.read_m(0xDB01), # ab按键对应的物品，参见物品表
            #"ItemB": self.read_m(0xDB00)
        }
        return observation
    """
    def _get_obs(self):
        cur_screen = self.preprocess_for_rl()
        return {
            "screen": np.array(cur_screen, dtype=np.uint8),           # (72,80,1)
            "health": np.array([self.cur_health], dtype=np.uint8),    # (1,)
            "agent_pos": np.array(self._get_pos(), dtype=np.int16)    # (2,)
        }

    def _get_info(self):
        """获取额外的游戏信息（针对不同房间设置）目前由于直接在单个房间中训练暂时不用太担心"""
        #room = self.read_m(0xDBAE)
        info = {
            "goal" : False,
            "room" : self.cur_room 
        }
        if self.check_goal():
            info = {
                "goal" : True,
                "room" : self.cur_room
            }
        return info
    
    def _get_pos(self):
        """获取当前角色所处的位置信息"""
        sprite = self.pyboy.get_sprite(2)
        x = sprite.x
        y = sprite.y
        return (x, y)
    
    def _get_tile(self):
        """将像素坐标粗略映射到 tile 网格坐标。GameBoy 典型 tile 为 8x8。"""
        x, y = self._get_pos()
        # 对异常/负值做保护
        try:
            tile_x = int(max(0, x) // 8)
            tile_y = int(max(0, y) // 8)
        except Exception:
            tile_x, tile_y = 0, 0
        return tile_x, tile_y
    
    def _reset_state(self):
        """检测角色目前的状态"""
        sprite = self.pyboy.get_sprite(2)
        if sprite.y == -16:
            if self.is_dead():
                return True
            else:
                return False
    
    def run_action(self, action):
        """执行特定的动作操作"""
        #TODO：这里直接使用button模拟按钮按下操作，后续可能需要调整为按下+释放等更精细的动作控制
        #self.pyboy.button(actions[action])
        self.pyboy.send_input(self.valid_actions[action])
        self.pyboy.tick(8)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(8)

    def is_hurt(self):
        if not isinstance(self.cur_health, (int, float)) or not isinstance(self.pre_health, (int, float)):
            return 0  # 无法计算，默认不受伤害
        
        if self.cur_health < self.pre_health:
            return self.cur_health - self.pre_health #XXX 在计算reward时乘一个折扣系数即可
        
        else:
            return 0
        
    def outside(self):
        if self.out_side >= 100:
            self.out_side = 0
            #self.reset() #长时间逗留在外则需要重新开始
            return True

        if self.cur_room != self.goal_room:
            self.out_side += 1
        else:
            self.out_side = 0
        
        return False
        
    def step(self, action):
        assert self.action_space.contains(action), "Invalid action!"
        self.cur_step += 1

        #self.pre_room = self.cur_room # 记录行动前的房间号
        
        self.pre_health = self.cur_health

        self.run_action(action)

        self.cur_health = self.read_m(0xDB5A)

        self.cur_room = self.read_m(0xDBAE)

        truncated = False

        new_reward,truncated = self.calculate_reward()

        self.visited_rooms.add(self.cur_room) # 将当前房间放入已访问房间中

        self.reward += new_reward

        observation = self._get_obs()

        info = self._get_info()

        done = self.is_done()

        #self.outside()

        # 长时间执行无意义动作则增加截断
        if(self.cur_step >= MAX_STEPS):
            truncated = True

        return observation, new_reward, done, truncated, info
    
    def check_visited(self, room_id):
        """检查当前房间是否被访问过"""
        return room_id in self.visited_rooms
    """
    def check_goal_59(self):
        # 针对59号房间的任务专门设置
        room_id = self.read_m(0xDBAE)
        self.cur_room = room_id
        if self.check_visited(room_id):
            self.pre_room = self.cur_room
        else:
            self.pre_room = 
            
    """
            
    def check_goal(self):
        """检查当前房间的目标是否完成"""
        #TODO 其他房间的奖励设置
        goal = self.room_goals.get(self.goal_room, None)
        if goal == "leave current room":
            #if self.check_visited(self.cur_room):
                #return True
                #TODO 这一块逻辑有误，需要重构
            if self.cur_room != 59:
                return True
            
        elif goal == "kill enemy and get key":
            if self.read_m(0xDBD0) >= 1:
                return True
            
        elif goal == "kill turtle,push button and open box":
            # XXX 目前暂定目标是拿到钥匙
            if self.read_m(0xDBD0) >= 1:
                return True

        return False
    
    def is_done(self):
        """判断游戏是否结束，或达成阶段性目标"""
        if self.check_goal():
            return True
        
        if self.is_dead():
            return True
        
        return False

    def is_dead(self):
        """判断角色是否死亡，如果死亡则这轮游戏结束"""
        if self.read_m(0xDB5A) == 0:
            return True

    def close(self):
        self.pyboy.stop()

    def read_m(self, addr): # 内存读取辅助函数
        return self.pyboy.memory[addr]
    
    def check_item(self,item = 0x0A):# 默认寻找羽毛
        """检查背包中是否有特定物品"""
        for addr in range(0xDB00, 0xDB0B):
            if self.read_m(addr) == item:
                return True
        
        return False
    
    def get_distance(self):
        room_id = self.goal_room
        x,y = self._get_pos()
        # TODO： 目前只完成了58号房间的distance，其他房间还未补充
        if room_id == 58:
            return abs(34 - x) + abs(45 - y)
        return 0
    
    def calculate_rupees(self):
        self.cur_rupee = self.read_m(0xDB5E)#要注意一下地址到底是多少
        if self.cur_rupee > self.pre_rupee:
            self.pre_rupee = self.cur_rupee
            return True
        return False

    def calculate_keys(self):
        """检测是否新获得钥匙，0xDBD0 背包钥匙计数。"""
        self.cur_keys = self.read_m(0xDBD0)
        if self.cur_keys > self.pre_keys:
            self.pre_keys = self.cur_keys
            return True
        return False

    def detect_button_press(self):
        """近似按钮踩下：位于按钮区域并连续驻留若干步。"""
        cfg = self.button_area_map.get(int(self.goal_room)) or self.button_area_map.get(int(self.cur_room))
        if not cfg:
            self.button_dwell = 0
            return False
        x1, y1, x2, y2 = cfg.get("area", [0, 0, -1, -1])
        dwell_need = int(cfg.get("dwell", 20))
        x, y = self._get_pos()
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.button_dwell += 1
            if self.button_dwell >= dwell_need:
                self.button_pressed = True
                return True
        else:
            self.button_dwell = 0
        return False

    def calculate_reward(self):
        """计算当前的奖励函数"""
        # 选择奖励权重配置（不拆任务，仅用作加权）
        profile_key = self.task_name if self.task_name in getattr(self, 'reward_profiles', {}) else 'default'
        weights = getattr(self, 'reward_profiles', {}).get(profile_key, self.reward_profiles['default'])
        # 允许通过 task_params 重写特定权重
        rw_override = (self.task_params or {}).get('reward_weights', {}) if hasattr(self, 'task_params') else {}
        if rw_override:
            weights = {**weights, **rw_override}

        reward = 0
        done = False
        if self.is_dead():
            reward -= float(weights.get('death_penalty', 1.0))

        #if self.is_hurt() != 0:
        reward += float(weights.get('hurt_coef', 0.01)) * self.is_hurt()

        # 击败敌人（近似）：卢比增长
        if self.calculate_rupees():
            reward += float(weights.get('rupee_reward', 1.0))
        # 拿到钥匙：给予最高奖励，并判为任务成功（终止本回合）
        if self.calculate_keys():
            reward += float(weights.get('get_key_reward', 20.0))
            done = True

        if self.check_goal():
            reward += float(weights.get('goal_reward', 10.0))
        else:
            if self.cur_room != self.goal_room:
                reward -= float(weights.get('room_penalty', 0.001))
            else:
                reward -= float(weights.get('distance_coef', 0.0001)) * self.get_distance()

        # 探索奖励：仅在目标房间内给予微小奖励，避免外房间刷分
        if int(self.cur_room) == int(self.goal_room):
            tile_x, tile_y = self._get_tile()
            tile_key = (int(self.cur_room), tile_x, tile_y)
            if tile_key not in self.visited_tiles:
                self.visited_tiles.add(tile_key)
                reward += float(weights.get('explore_bonus', self.explore_bonus))
       # 踩下按钮（近似）
        if not self.button_pressed and self.detect_button_press():
            reward += float(weights.get('press_button_reward', 3.0))

        if self.outside():
            reward -= float(weights.get('outside_penalty', 0.1))
            done = True
        # 子任务奖励叠加
        sub_reward, sub_done = self._compute_subtask_reward()
        reward += sub_reward
        if self.terminate_on_subtask and sub_done:
            done = True
        return reward, done
        
    
    def _compute_subtask_reward(self):
        """根据当前激活的子任务给予奖励与结束信号。"""
        if not self.task_name:
            return 0.0, False

        name = self.task_name
        params = self.task_params or {}
        sub_reward = 0.0
        sub_done = False

        # 1) 拿到钥匙
        if name == 'get_key':
            has_key = self.read_m(0xDBD0) >= int(params.get('min_keys', 1))
            if has_key:
                sub_reward += float(params.get('reward', 5.0))
                sub_done = True

        # 2) 探索指定覆盖率
        elif name == 'explore_tiles':
            target_tiles = int(params.get('target_tiles', 50))
            if len(self.visited_tiles) >= target_tiles:
                sub_reward += float(params.get('reward', 2.0))
                sub_done = True
         # 3) 接近/到达指定区域（像素坐标，曼哈顿距离）
        elif name == 'reach_area':
            target = params.get('target_xy', [34, 45])
            radius = float(params.get('radius', 6.0))
            tx, ty = int(target[0]), int(target[1])
            x, y = self._get_pos()
            dist = abs(tx - x) + abs(ty - y)
            sub_reward += -0.0001 * dist  # shaping
            if dist <= radius:
                sub_reward += float(params.get('reward', 2.0))
                sub_done = True

        # 4) 按压地板按钮（用区域停留近似）
        elif name == 'press_button':
            area = params.get('area_xyxy', [28, 36, 40, 48])  # [x1,y1,x2,y2]
            dwell = int(params.get('dwell_steps', 20))
            x, y = self._get_pos()
            x1, y1, x2, y2 = area
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.task_dwell_steps += 1
                sub_reward += 0.0005
                if self.task_dwell_steps >= dwell:
                    sub_reward += float(params.get('reward', 3.0))
                    sub_done = True
            else:
                self.task_dwell_steps = 0
        # 5) 击杀敌人（用卢比上涨作为近似信号）
        elif name == 'kill_enemy':
            if self.calculate_rupees():
                sub_reward += float(params.get('reward', 3.0))
                sub_done = True

        return sub_reward, sub_done