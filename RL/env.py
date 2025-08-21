import numpy as np
import random
from pyboy import PyBoy
import gym
from gym import spaces
from pyboy.utils import WindowEvent

game_file = "RL\game_state\Link's awakening.gb"
#TODO ：在后续为每个房间都保存相应的state文件
save_file = "RL\game_state\Link's awakening.gb.state"

mode = "human"
actions = ["","a","b","left","right","up","down"]

class Zelda_Env(gym.Env):
    def __init__(self, game_file, save_file, mode):
        """
        初始化游戏环境
        """
        super().__init__()
        self.pyboy = PyBoy(game_file, sound_emulated = False)
        try:
            with open(save_file, "rb") as f:
                self.pyboy.load_state(f)
        except FileNotFoundError:
            print("No existing save file, starting new game")
        
        """
        游戏状态参数设置
        """
        self.max_health = self.read_m[0xDB5B]
        self.health = None
        # 当前所处的迷宫房间号
        self.room = None
        self.zelda = self.pyboy.game_wrapper

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
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        self.screen = self.pyboy.screen
        screen_shape = self.screen.ndarray.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_shape[0],screen_shape[1]), dtype=np.uint8)
        """
        训练参数设置
        """ 
        self.reward = 0
        self.step = None
        self.episode = 0

    # 游戏画面渲染设置
    def render(self, mode = "human"):
        if mode == "human":
            self.pyboy.render_screen()

    def _get_obs(self):
        """
        读取游戏当前状态，并将其转换成易于读取的格式
        """
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
        """
        获取额外的游戏信息（针对不同房间设置）
        """
        room = self.read_m(0xDBAE)

        info = {

        }
        return info
    
    def _get_pos(self):
        """
        获取当前角色所处的位置信息
        """
        sprite = self.pyboy.get_sprite(2)
        x = sprite.x
        y = sprite.y
        return (x, y)

    def reset(self):
        self.step = 0
        self.episode += 1
        self.reward = 0
        """
        重置游戏状态
        """
        self.pyboy.stop()
        self.pyboy = PyBoy(game_file, sound_emulated = False)
        try:
            with open(save_file, "rb") as f:
                self.pyboy.load_state(f)
        except FileNotFoundError:
            print("No existing save file, starting new game")
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def close(self):
        self.pyboy.stop()

    def read_m(self, addr): # 内存读取辅助函数
        return self.pyboy.memory[addr]
    
    def run_action(self, action):
        """
        执行特定的动作操作
        """
        #TODO：这里直接使用button模拟按钮按下操作，后续可能需要调整为按下+释放等更精细的动作控制
        #self.pyboy.button(actions[action])
        #self.pyboy.tick(10) #每个动作的执行时间
        self.pyboy.send_input(self.valid_actions[action])
        self.pyboy.tick(8)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(8)
    
    def step(self, action):
        self.step += 1
        assert self.action_space.contains(action), "Invalid action!"
        
        self.run_action(action)

        new_reward = self.calculate_reward()
        self.reward += new_reward

        observation = self._get_obs()

        info = self._get_info()

        done = self.is_done()

        return observation, new_reward, done, info
    
    def is_done(self):
        """
        判断游戏是否结束，血量清零或者达成目标
        """
        if self.read_m(0xDB5A) <= 0:
            print("game over, you are died!")
            return True
        return False

    def check_item(self,item = 0x0A):# 默认寻找羽毛
        """
        检查背包中是否有特定物品
        """
        for addr in range(0xDB00, 0xDB0B):
            if self.read_m(addr) == item:
                return True
        
        return False
    
    def calculate_reward(self):
        """
        计算当前的奖励函数
        """
        # TODO

