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

#actions = ["", "a", "b", "left", "right", "up", "down", "start", "select"]


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
#actions = ["","left","right","up","down"]
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
        super().__init__()
        # 初始化PyBoy环境
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
        # 执行动作，动作选择来自action参数
        assert self.action_space.contains(action), "Invalid action!"
        current_health = self.pyboy.memory[0xDB5A]
        current_a,current_b = self.get_distance()
        if action != 0:
            self.pyboy.button(actions[action])

        self.pyboy.tick(5)

        # 计算奖励（来源于reward函数）
        reward = self._calculate_reward()

        """
        不能让角色走出房间😡
        """
        if(self.pyboy.memory[0xDBAE] != 51):
            self.reset()
            reward -= 0.1
        
        a,b = self.get_distance()

        x,y = self.get_position()

        if self.flag:
            """
            此时需要让ai前往按钮位置
            """
            reward += (- a + current_a)/10
            """
            新增阶段性奖励
            """
            if self.station == 0:
                if(y < 40):
                    reward += (- b + current_b)/10
                if(abs(x - 125) < 5 and abs(y - 80) < 10):
                    reward += 1000
                    self.station = 1
                reward += (200 - (abs(x -125) + abs(y - 80)))/ 100
            elif self.station == 1:
                if (abs(x - 125) < 10 and abs(y - 10) < 10):
                    reward += 1000
                    self.station = 2
                reward += (200 - (abs(x -125) + abs(y - 10)))/ 100
                            
        else:
            reward += (- b + current_b)/10
        
        # 对于扣血操作加以惩罚
        new_health = self.pyboy.memory[0xDB5A]
        reward -= (current_health - new_health) * 0.01

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

    def get_distance(self):
        """获取人物的位置"""
        x, y = self.get_position()
        distance_one = abs(x - 80) + abs(y - 45) # 与按钮的距离
        distance_two = abs(x - 128) + abs(y - 42) # 与宝箱的距离
        return distance_one, distance_two
    
    def _calculate_reward(self):
        """计算奖励：基于屏幕变化"""
        # TODO:补全奖励函数的设置
        reward = 0

        x,y = self.get_position()
        if(abs(x - 80) < 5 and abs(y - 45) < 5 and self.flag):
            self.flag = False
            reward += 2000
        
        distance_one, distance_two = self.get_distance()

        if(x,y) == (72,112):
            reward -= 1
        
        if(self.pyboy.memory[0xDBD0] == 1):
            reward += 2000
        return reward

    def _get_observation(self):
        """获取游戏区域的画面"""
        screen_data = self.screen.ndarray
        grayscale = np.dot(screen_data[...,:3], [0.2989, 0.5870, 0.1140])
        return grayscale.astype(np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        """
        重置游戏阶段标志
        """
        self.flag = True
        self.station = 0
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
    

# 自定义回调函数用于记录 reward
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_reward = 0

    def _on_step(self) -> bool:
        # 每一步累加 reward
        reward = self.locals["rewards"][0]
        self.episode_reward += reward

        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
        return True

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

# 运行测试，先测试随机策略下环境能否正常运行
"""
if __name__ == "__main__":
    env = ZeldaPyBoyEnv("Link's awakening.gb", "Link's awakening.gb.state")
    
    obs, info = env.reset()
    
    for i in range(1000):
        env.pyboy.tick(1)
        if i % 10 == 0:
            action = random.randint(0, len(actions) - 1)  # 随机选择动作
            obs, reward, done, _, _ = env.step(action)
        if done:
            break
    
    env.close()
"""
# 训练函数
def train():
    #env = ZeldaPyBoyEnv(ROM_PATH, STATE_PATH)
    """
    防止AI陷入无意义行动中增加最大步数限制
    """
    env = TimeLimit(ZeldaPyBoyEnv(ROM_PATH, STATE_PATH), max_episode_steps=2000)

    env = Monitor(env)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_zelda/", device='cpu')
    # 创建 reward 日志器
    reward_callback = RewardLoggerCallback()
    # 训练
    model.learn(total_timesteps=100000, callback=reward_callback)
    # 绘图
    plot_rewards(reward_callback.episode_rewards)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"Mean reward: {mean_reward}, Std: {std_reward}")

    model.save("RL_model/ppo_zelda")

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
    train()
    #test()