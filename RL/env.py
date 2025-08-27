import numpy as np
import random
from pyboy import PyBoy
import gym
from gym import spaces
from pyboy.utils import WindowEvent

game_file = "RL\game_state\Link's awakening.gb"
#TODO ：在后续为每个房间都保存相应的state文件
save_file = "RL\game_state\Link's awakening.gb.state"

#mode = "human"
actions = ["","a","b","left","right","up","down"]

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
        self.max_health = self.read_m[0xDB5B]
        self.pre_health = self.read_m[0xDB5A]
        self.cur_health = self.pre_health
        # 当前所处的迷宫房间号
        self.pre_room = None
        self.cur_room = self.read_m[0xDBAE]
        self.visited_rooms = set()
        self.zelda = self.pyboy.game_wrapper
        # 设置不同房间的任务目标
        self.room_goals = {
            59: "leave current room", # 59号房间是迷宫入口房间
            58: "kill enemy and get key" # 迷宫入口左侧房间，有两个乌龟怪物
        }

        """动作空间设定"""
        # 定义动作空间和观察空间
        self.action_space = spaces.Discrete(len(actions))
        # 更精细的动作控制则需要使用如下的形式
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
        self.screen = self.pyboy.screen
        screen_shape = self.screen.ndarray.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_shape[0],screen_shape[1]), dtype=np.uint8)
        """训练参数设置""" 
        self.reward = 0
        self.cur_step = None
        self.episode = 0

    # 游戏画面渲染设置
    def render(self, mode = "human"):
        if mode == "human":
            self.pyboy.render_screen()

    def _get_obs(self):
        """读取游戏当前状态，并将其转换成易于读取的格式"""
        self.health = self.read_m[0xDB5A]
        self.max_health = self.read_m[0xDB5B]
        
        observation = {
            "agent_pos": self._get_pos(),
            "health" : self.health,
            "max_health":self.max_health,
            "game_area": self.pyboy.game_area(),
            "key" : self.read_m(0xDBD0), # 读取钥匙数量，1表示成功拿到钥匙
            "room" :self.read_m(0xDBAE), # 当前所处的房间位置
            "ItemA": self.read_m(0xDB01), # ab按键对应的物品，参见物品表
            "ItemB": self.read_m(0xDB00)
        }
        return observation
    
    
    def _get_info(self):
        """获取额外的游戏信息（针对不同房间设置）目前由于直接在单个房间中训练暂时不用太担心"""
        #room = self.read_m(0xDBAE)

        info = {

        }
        return info
    
    def _get_pos(self):
        """获取当前角色所处的位置信息"""
        sprite = self.pyboy.get_sprite(2)
        x = sprite.x
        y = sprite.y
        return (x, y)

    def reset(self):
        self.cur_step = 0
        self.episode += 1
        self.reward = 0

        # 重置走过的房间编号
        self.pre_room = self.read_m[0xDBAE]
        self.cur_room = self.pre_room
        self.visited_rooms = set()

        """重置游戏状态
        self.pyboy.stop()
        self.pyboy = PyBoy(game_file, sound_emulated = False)
        try:
            with open(save_file, "rb") as f:
                self.pyboy.load_state(f)
        except FileNotFoundError:
            print("No existing save file, starting new game")
        """
        #这里采用更方便的方式，及直接使用stateload来重置游戏
        self.pyboy.send_input(WindowEvent.STATE_LOAD) 
        self.pyboy.tick(1)
        #self.pyboy.send_input(WindowEvent.STATE_LOAD)
        
        # 重置其他状态参数
        self.pre_health = self.read_m[0xDB5A]
        self.cur_health = self.pre_health

        self.pre_room = None
        self.cur_room = self.read_m[0xDBAE]

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def close(self):
        self.pyboy.stop()

    def read_m(self, addr): # 内存读取辅助函数
        return self.pyboy.memory[addr]
    
    def run_action(self, action):
        """执行特定的动作操作"""
        #TODO：这里直接使用button模拟按钮按下操作，后续可能需要调整为按下+释放等更精细的动作控制
        #self.pyboy.button(actions[action])
        self.pyboy.send_input(self.valid_actions[action])
        self.pyboy.tick(8)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(8)

    def is_hurt(self):
        if self.cur_health < self.pre_health:
            return True
        
    def step(self, action):
        assert self.action_space.contains(action), "Invalid action!"
        self.cur_step += 1

        self.pre_room = self.cur_room # 记录行动前的房间号

        self.run_action(action)
        self.cur_room = self.read_m(0xDBAE)

        new_reward = self.calculate_reward()

        self.visited_rooms.add(self.room) # 将当前房间放入已访问房间中

        self.reward += new_reward

        observation = self._get_obs()

        info = self._get_info()

        done = self.is_done()

        return observation, new_reward, done, info
    
    def check_visited(self, room_id):
        """检查当前房间是否被访问过"""
        return room_id in self.visited_rooms
    
    def check_goal(self):
        """检查当前房间的目标是否完成"""
        goal = self.room_goads.get(self.pre_room, None)
        if goal == "leave current room":
            if self.check_visited(self.cur_room):
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

    def check_item(self,item = 0x0A):# 默认寻找羽毛
        """检查背包中是否有特定物品"""
        for addr in range(0xDB00, 0xDB0B):
            if self.read_m(addr) == item:
                return True
        
        return False
    
    def calculate_reward(self):
        """计算当前的奖励函数"""
        # TODO
        reward = 0
        if self.is_dead():
            reward += -5
        if self.is_hurt():
            reward += -1
        if self.check_goal():
            reward += 10
        return reward