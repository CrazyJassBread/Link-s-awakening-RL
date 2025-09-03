import numpy as np
import random
from pyboy import PyBoy
import gymnasium as gym
from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from skimage.transform import downscale_local_mean

# 设置最大步数限制
TOTAL_STEPS = 1000000
MAX_STEPS = 1000

game_file = "RL\game_state\Link's awakening.gb"
#XXX ：为每个房间都保存相应的state文件,当前任务是房间58
save_file = "RL\game_state\Room_58.state"

class Zelda_Env(gym.Env):
    def __init__(self, game_file, save_file):

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
        # 当前所处的迷宫房间号
        self.goal_room = self.read_m(0xDBAE)
        self.cur_room = self.read_m(0xDBAE)
        self.out_side = 0 # XXX 计算agent脱离目标房间的时间
        # XXX 新增房间目标完成检测
        self.cur_goal = False
        self.visited_rooms = set()
        #self.zelda = self.pyboy.game_wrapper
        # 设置不同房间的任务目标
        self.room_goals = {
            59: "leave current room", # 59号房间是迷宫入口房间
            58: "kill enemy and get key" # 迷宫入口左侧房间，有两个乌龟怪物
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

        self.pre_health = self.read_m(0xDB5A)
        self.cur_health = self.pre_health

        #self.goal_room = self.read_m(0xDBAE)
        #self.cur_room = self.read_m(0xDBAE)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    # 游戏画面渲染设置
    def render(self, mode = "human"):
        if mode == "human":
            self.pyboy.render_screen()
    
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
        if room_id == 58:
            return abs(34 - x) + abs(45 - y)
        return 0
    
    def calculate_reward(self):
        """计算当前的奖励函数"""
        # TODO
        reward = 0
        done = False
        if self.is_dead():
            reward += -1

        #if self.is_hurt() != 0:
        reward += 0.01 * self.is_hurt()

        if self.check_goal():
            reward += 10
        else:
            if self.cur_room != self.goal_room:
                reward -= 0.001
            else:
                reward -= 0.0001 * self.get_distance()

        if self.outside():
            reward -= 0.1
            done = True
        return reward, done