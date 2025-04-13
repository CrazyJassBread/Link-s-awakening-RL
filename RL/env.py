import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
import numpy as np
import random
from pyboy import PyBoy

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# 绘制学习曲线
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

# 禁用声音输出
import os
os.environ["SDL_AUDIODRIVER"] = "dummy"  # 禁用声音输出

DEBUG = False

"""
PPO模型参数设置
"""
LR = 0.0003
BATCH_SIZE = 64
GAMMA = 0.99
N_STEPS = 2048
NORMALIZE_ADVADTAGE = False
"""
游戏ROM、STATE地址
"""
ROM_PATH = "game_state/Link's awakening.gb"
STATE_PATH = "game_state/Link's awakening.gb.state"

actions = ["","a","b","left","right","up","down"]
class ZeldaPyBoyEnv(gym.Env):
    def __init__(self, rom_path, state_path):
        self.debug = DEBUG
        """
        游戏阶段标志旗帜FLAG
        """
        self.flag = True
        """
        设置station标志游戏进行的阶段 来激活不同的奖励函数
        """
        self.station = 0
        self.pre_station = 0
        """
        设置room参数 判断ai当前的房间位置
        """
        self.room = True
        # 打印奖励
        self.count = 0
        self.reward = 0
        # 初始化环境
        super().__init__()
        self.pyboy = PyBoy(rom_path, sound=False)
        if state_path:
            with open(state_path, "rb") as f:
                self.pyboy.load_state(f)

        self.pyboy.set_emulation_speed(0 if not self.debug else 1)

        self.zelda = self.pyboy.game_wrapper

        # 定义动作空间和观察空间
        self.action_space = spaces.Discrete(len(actions))  

        self.screen = self.pyboy.screen
        screen_shape = self.screen.ndarray.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_shape[0],screen_shape[1]), dtype=np.uint8)
        
    def step(self, action):
        self.count += 1
        # 执行动作，动作选择来自action参数
        pre_x, pre_y = self.get_position()
        assert self.action_space.contains(action), "Invalid action!"
        current_health = self.pyboy.memory[0xDB5A]
        """
        通过station的数值判断当前阶段，并且读取相应的位置地址
        """
        reward = self._calculate_reward()
        if self.station == 0: # (125,80)or(25,80)
            x,y = self.get_position()
            if x > 72:
                pre_dist = self.get_distance(120,80)
            else:
                pre_dist = self.get_distance(25,80)
        elif self.station == 1: # (125,45)or(25,45)
            x,y = self.get_position()
            if x > 72:
                pre_dist = self.get_distance(120,45)
                pre_dist_a = self.get_distance(142,45)
            else:
                pre_dist = self.get_distance(25,45)
                pre_dist_a = self.get_distance(2,45)
        elif self.station == 2: # (125, 10)or(25, 10)
            x,y = self.get_position()
            if x > 72:
                pre_dist = self.get_distance(120, 10)
                pre_dist_a = self.get_distance(142,45)
            else:
                pre_dist = self.get_distance(25, 10)
                pre_dist_a = self.get_distance(2,45)
        elif self.station == 3: # (80, 10),引导人物前往按钮上方
            x,y = self.get_position()    
            pre_dist = self.get_distance(80, 10)
        elif self.station == 4: # (80, 45),引导人物前往按钮
            x,y = self.get_position()    
            pre_dist = self.get_distance(80, 45)
        
        elif self.station == 5:
            pre_dist = self.get_distance(80,10)
        elif self.station == 6:
            pre_dist = self.get_distance(124,10)
        elif self.station == 7:
            pre_dist = self.get_distance(124,45)


        if action != 0:
            self.pyboy.button(actions[action])
        self.pyboy.tick(10)
        
        x,y = self.get_position()
        """
        不能让角色走出房间😡
        """
        if(self.room):
            # 计算奖励（来源于reward函数）
            if(self.pyboy.memory[0xDBAE] != 51):
                self.pyboy.tick(10)
                if self.station != -1:
                    self.pre_station = self.station
                    self.station = -1
                reward -= 0.2
                self.room = False
            else:
                if self.station == -1:
                    self.station = self.pre_station


            """
            此时需要让ai前往按钮位置
                
            新增阶段性奖励
            """
            if self.station == 0:
                if x > 72:
                    cur_dist = self.get_distance(120, 80)
                    reward += (pre_dist - cur_dist) / 40
                    if action == 4 and pre_x != x:
                        reward += 0.005
                else:
                    cur_dist = self.get_distance(25, 80)
                    reward += (pre_dist - cur_dist) / 40
                    if action == 3 and pre_x != x:
                        reward += 0.005

            elif self.station == 1:
                if action == 5 and pre_y != y:
                    reward += 0.005
                if x > 72:
                    cur_dist = self.get_distance(120, 45)
                    reward += (pre_dist - cur_dist) / 30
                    cur_dist_a = self.get_distance(142,50)
                    reward -= (pre_dist_a - cur_dist_a) / 40
                else:
                    cur_dist = self.get_distance(25, 45)
                    reward += (pre_dist - cur_dist) / 30
                    cur_dist_a = self.get_distance(2,50)
                    reward -= (pre_dist_a - cur_dist_a) / 40

            elif self.station == 2:
                if action == 5 and pre_y != y:
                    reward += 0.005
                if x > 72:
                    cur_dist = self.get_distance(120, 10)
                    reward += (pre_dist - cur_dist) / 20
                    cur_dist_a = self.get_distance(142,50)
                    reward -= (pre_dist_a - cur_dist_a) / 40
                else:
                    cur_dist = self.get_distance(25, 10)
                    reward += (pre_dist - cur_dist) / 20
                    cur_dist_a = self.get_distance(2,50)
                    reward -= (pre_dist_a - cur_dist_a) / 40

            elif self.station == 3:
                if(x > 72):
                    if action == 3 and pre_x != x:
                        reward += 0.01
                else:
                    if action == 4 and pre_x != x:
                        reward += 0.01
                if x == pre_x:
                    reward -= 0.005
                cur_dist = self.get_distance(80, 10)
                reward += (pre_dist - cur_dist) / 10

            elif self.station == 4:
                cur_dist = self.get_distance(80, 45)
                reward += (pre_dist - cur_dist) / 10
            
            elif self.station == 5:
                cur_dist = self.get_distance(80,10)
                reward += (pre_dist - cur_dist) / 10

            elif self.station == 6:
                cur_dist = self.get_distance(124,10)
                reward += (pre_dist - cur_dist) / 10

            elif self.station == 7:
                cur_dist = self.get_distance(124,45)
                reward += (pre_dist - cur_dist) / 10


            # 对于扣血操作加以惩罚
            new_health = self.pyboy.memory[0xDB5A]
            reward -= (current_health - new_health) * 0.05
            if(self.count % 100 == 0):
                self.count = 0
                print(self.reward,self.station)
        else:
            
            reward -= 0.01
            if(self.count % 100 == 0):
                self.count = 0
                print(self.reward,self.station)
        self.reward += reward

        # 判断游戏是否结束
        done = self.game_over()

        observation = self._get_observation()

        return observation, reward, done, False, {}

    def game_over(self):
        # 暂时通过是否拿到钥匙和人物血量来判断游戏是否结束
        # TODO:后续可以通过修改不同阶段的返回值来判断游戏阶段
        if self.pyboy.memory[0xDBD0] == 1:
            return True
        elif self.pyboy.memory[0xDB5A] == 0:
            return True
        else:
            return False

    """
    设置距离计算函数，用于提取人物的距离和计算人物与目标的距离
    """
    def get_position(self):
        
        sprite = self.pyboy.get_sprite(2)
        x = sprite.x
        y = sprite.y
        return x, y

    def get_distance(self,a,b):
        """获取人物的位置"""
        x, y = self.get_position()
        distance = abs(x - a) + abs(y - b) # 计算与（a，b）的距离
        return distance
    
    def _calculate_reward(self):
        """
        计算奖励, 对于踩下按钮和打开宝箱给出高额奖励,并且对于阶段奖励的完成给出状态切换
        """
        # TODO:补全奖励函数的设置
        reward = 0
        x,y = self.get_position()
        if self.station == 0:
            if(x > 72):
                dist = self.get_distance(120,80)
                if(dist < 8 or (y < 60 and y > 10)):
                    self.station = 1
                    reward += 1
            else:
                dist = self.get_distance(25,80)
                if(dist < 8 or (y < 60 and y > 10)):
                    self.station =1
                    reward += 1
        elif self.station == 1:
            if(x > 72):
                dist = self.get_distance(120,45)
                if(dist < 8 or (y < 45 and y > 10)):
                    self.station = 2
                    reward += 3
            else:
                dist = self.get_distance(25,45)
                if(dist < 8 or (y < 45 and y > 10)):
                    self.station =2
                    reward += 3
        elif self.station == 2:
            if(x > 72):
                dist = self.get_distance(120,10)
                if(dist < 10 or y < 10):
                    self.station = 3
                    reward += 10
            else:
                dist = self.get_distance(25,10)
                if(dist <10 or y < 10):
                    self.station = 3
                    reward += 10
        elif self.station == 3:  
            dist = self.get_distance(80,10)
            if(dist < 8):
                self.station = 4
                reward += 10
        
        if(abs(x - 80) < 4 and abs(y - 45) < 4):
            reward += 200
            self.station = 5

        if self.station == 5:
            dist = self.get_distance(80,10)
            if(dist < 8 or y < 10):
                self.station = 6

        elif self.station == 6:
            dist = self.get_distance(124,10)
            if(dist < 8 or x > 120):
                self.station = 7
        elif self.station == 7:
            dist = self.get_distance(124,45)
            if(dist < 8 or y > 45):
                self.station = 7
        
        if(self.pyboy.memory[0xDBD0] == 1):
            reward += 400
        return reward

    def _get_observation(self):
        """获取游戏区域的画面"""
        screen_data = self.screen.ndarray
        grayscale = np.dot(screen_data[...,:3], [0.2989, 0.5870, 0.1140])
        return grayscale.astype(np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        """
        重置游戏阶段
        """
        self.pre_station = 0
        self.station = 0
        self.reward = 0
        self.room = True
        self.count = 0
        # 重新加载游戏环境
        self.pyboy.stop()
        self.pyboy =  PyBoy(ROM_PATH, sound=False)
        if STATE_PATH:
            with open(STATE_PATH, "rb") as f:
                self.pyboy.load_state(f)
        self.pyboy.set_emulation_speed(0 if not self.debug else 1)
        self.pyboy.tick(1)
        # 返回值
        return self._get_observation(), {}

    def close(self):
        self.pyboy.stop()
    
    def render(self, mode='human'):
        if mode == 'human':
            self.pyboy.render_screen()
    
# 自定义回调函数用于记录 rewardclass RewardLoggerCallback(BaseCallback):
class RewardLoggerCallback(BaseCallback):
    def __init__(self, save_freq=5, save_dir="Reward_image", verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # 检查是否是 episode 结束
        if self.locals.get("dones") and any(self.locals["dones"]):
            infos = self.locals.get("infos")
            for info in infos:
                if "episode" in info:
                    reward = info["episode"]["r"]
                    self.episode_rewards.append(reward)

                    # 每 save_freq 次保存一次图像
                    if len(self.episode_rewards) % self.save_freq == 0:
                        self.plot_and_save(len(self.episode_rewards))

        return True

    def plot_and_save(self, idx):
        plt.figure(figsize=(10, 4))
        plt.plot(self.episode_rewards, label='Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Reward Curve (Episode {idx})')
        plt.legend()
        path = os.path.join(self.save_dir, f"reward_plot_{idx:04d}.png")
        plt.savefig(path)
        plt.close()

def plot_rewards(rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Episode Reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Zelda PPO Training Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reward_curve.png") 
    plt.show()

# 训练函数
def train():
    #env = ZeldaPyBoyEnv(ROM_PATH, STATE_PATH)
    """
    防止AI陷入无意义行动中增加最大步数限制
    """
    env = TimeLimit(ZeldaPyBoyEnv(ROM_PATH, STATE_PATH), max_episode_steps=2000)

    env = Monitor(env)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_zelda/", device='cpu')
    reward_callback = RewardLoggerCallback(save_freq=10, save_dir="images")

    # 训练
    try:
        model.learn(total_timesteps=500000, callback=reward_callback)
    finally:
        print("Saving model...")
        model.save("RL_model/ppo_zelda")
        print("Model saved!")
    env.close()
    # 绘图
    plot_rewards(reward_callback.episode_rewards)
    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    #print(f"Mean reward: {mean_reward}, Std: {std_reward}")

# 测试函数
def test():
    env = ZeldaPyBoyEnv(ROM_PATH, STATE_PATH)
    model = PPO.load("RL_model/ppo_zelda.zip", env=env)

    done = False
    obs, _ = env.reset()

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

    env.close()

if __name__ == "__main__":
    #train()
    test()