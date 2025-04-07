import gym
from gym import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from pathlib import Path

# #########################
# 向左走 + reward, 向右走 - reward, 保持大体方向
##########################
class ZeldaEnv(gym.Env):
    def __init__(self, rom_path, config=None):
        super(ZeldaEnv, self).__init__()

        # 配置初始化
        self.s_path = Path(config.get("session_path", "./"))  # 存储路径
        self.print_rewards = config.get("print_rewards", False)  # 是否打印奖励
        self.headless = config["headless"]  # 是否无头模式（不显示图形界面）
        self.init_state = config["init_state"]  # 初始状态文件路径
        self.save_video = config["save_video"]  # 是否保存视频
        self.fast_video = config.get("fast_video", False)  # 是否快速视频
        self.frame_stacks = 3  # 堆叠的帧数
        self.action_freq = config.get("action_freq", 4)  # 动作频率
        self.max_steps = config.get("max_steps", 10000000)  # 最大步骤数

        self.s_path.mkdir(exist_ok=True)  # 创建存储路径
        self.full_frame_writer = None  # 完整视频帧写入器
        self.model_frame_writer = None  # 模型视频帧写入器
        self.map_frame_writer = None  # 地图视频帧写入器

        self.reset_count = 0  # 重置计数器
        # 定义动作空间（允许的按钮按下动作）
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            # WindowEvent.PRESS_BUTTON_START,
        ]

        # 定义释放动作
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            # WindowEvent.RELEASE_BUTTON_START
        ]

        # 设置动作空间（动作数量与 valid_actions 一致）
        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.metadata = {"render.modes": []}  # 渲染模式
        self.reward_range = (0, 15000)  # 奖励范围
        self.cumulative_reward = 0  # 累计奖励初始化

        # 定义状态空间
        self.observation_space = spaces.Dict(
            {
                "health": spaces.Box(low=0, high=14, shape=(1,), dtype=np.uint8),  # 当前生命值（0-14颗心）
                "max_health": spaces.Box(low=0, high=14, shape=(1,), dtype=np.uint8),  # 最大生命值（最大14颗心）
                "rupees": spaces.Box(low=0, high=999, shape=(1,), dtype=np.uint16),  # 当前卢比数量
                "position": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),  # 当前坐标（位置）
                "inventory": spaces.MultiBinary(10),  # 库存物品（最多 10 个物品）
                "key_count": spaces.Box(low=0, high=99, shape=(1,), dtype=np.uint8),  # 钥匙数量
                "arrows": spaces.Box(low=0, high=99, shape=(1,), dtype=np.uint8),  # 弓箭数量
                "bombs": spaces.Box(low=0, high=99, shape=(1,), dtype=np.uint8),  # 炸弹数量
                "magic_powder": spaces.Box(low=0, high=99, shape=(1,), dtype=np.uint8),  # 魔法粉数量
                "map_progress": spaces.MultiBinary(256),  # 已访问的地图区域（每个区域为一个标志位）
                "secrets": spaces.Box(low=0, high=999, shape=(1,), dtype=np.uint16),  # 秘密贝壳数量
                "golden_leaves": spaces.Box(low=0, high=99, shape=(1,), dtype=np.uint8),  # 金叶数量
                "instruments": spaces.MultiBinary(8),  # 已获得的乐器（8个地牢）
            }
        )

        # 使用无头模式或SDL2窗口显示
        head = "null" if config["headless"] else "SDL2"

        self.pyboy = PyBoy(  # 初始化 PyBoy 模拟器
            rom_path,  # 游戏 ROM 路径
            window=head,
        )

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)  # 设置模拟器的仿真速度

        # 修改：新增杀敌计数和钥匙获取标志
        self.enemies_killed = 0
        self.key_obtained = False
        self.last_health = 1

        # 初始化敌人内存监控
        self.prev_enemy_memory = np.array([self.pyboy.memory[i] for i in range(0xD700, 0xD79C)])

    def reset(self, seed=None, options={}):
        self.seed = seed
        # 加载游戏的初始状态
        try:
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)  # 加载保存的游戏状态
        except FileNotFoundError:
            print(f"Warning: Game state file {self.init_state} not found, starting a new game.")

        # 初始化地图内存
        self.init_map_mem()

        # 初始化代理的状态统计
        self.agent_stats = []

        self.explore_map_dim = (8, 8)
        self.explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)

        self.recent_screens = np.zeros((144, 160, 3), dtype=np.uint8)
        self.recent_actions = np.zeros(self.frame_stacks, dtype=np.uint8)

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0

        # 初始化事件标志， 这个是？好像没用到
        self.base_event_flags = sum([
            self.bit_count(self.read_m(i))
            for i in range(0xD747, 0xD87E)  # 假设事件标志的地址范围
        ])

        self.current_event_flags_set = {}

        # 初始化进度奖励
        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()  # 现在是整数
        self.total_reward = self.progress_reward  # 直接赋值
        self.reset_count += 1

        # 重置杀敌计数和钥匙获取标志
        self.enemies_killed = 0
        self.key_obtained = False

        # 重置敌人内存监控
        self.prev_enemy_memory = np.array([self.pyboy.memory[i] for i in range(0xD700, 0xD79C)])

        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}

    def _get_obs(self):
        # 获取当前状态
        current_health = self.pyboy.memory[0xDB5A]  # 当前生命值
        max_health = self.pyboy.memory[0xDB5B]  # 最大生命值
        rupees = self.pyboy.memory[0xDB5D] * 256 + self.pyboy.memory[0xDB5E]  # 当前卢比数量
        position = self.pyboy.memory[0xDBAE]  # 当前在地牢中的位置

        # 其他游戏状态项
        inventory = [self.pyboy.memory[0xDB02 + i] for i in range(10)]  # 库存物品
        key_count = self.pyboy.memory[0xDBD0]  # 钥匙数量
        arrows = self.pyboy.memory[0xDB45]  # 弓箭数量
        bombs = self.pyboy.memory[0xDB4D]  # 炸弹数量
        magic_powder = self.pyboy.memory[0xDB4C]  # 魔法粉数量
        secrets = self.pyboy.memory[0xDB0F]  # 秘密贝壳数量
        golden_leaves = self.pyboy.memory[0xDB15]  # 金叶数量
        instruments = [self.pyboy.memory[0xDB65 + i] for i in range(8)]  # 已获得的乐器
        map_progress = [self.pyboy.memory[0xD800 + i] for i in range(256)]  # 已访问的地图区域

        state = {
            "health": np.array([current_health]),
            "max_health": np.array([max_health]),
            "rupees": np.array([rupees]),
            "position": np.array([position]),
            "inventory": np.array(inventory),
            "key_count": np.array([key_count]),
            "arrows": np.array([arrows]),
            "bombs": np.array([bombs]),
            "magic_powder": np.array([magic_powder]),
            "map_progress": np.array(map_progress),
            "secrets": np.array([secrets]),
            "golden_leaves": np.array([golden_leaves]),
            "instruments": np.array(instruments)
        }
        return state

    def _take_action(self, action):
        self.pyboy.send_input(self.valid_actions[action])
        self.pyboy.tick(self.action_freq)
        # 释放动作
        self.pyboy.send_input(self.release_actions[action])

        # 监控敌人内存变化
        current_enemy_memory = np.array([self.pyboy.memory[i] for i in range(0xD700, 0xD79C)])
        changed_indices = np.where(current_enemy_memory != self.prev_enemy_memory)[0]
        if len(changed_indices) > 0:
            print(f"Enemy memory changed at addresses: {[hex(0xD700 + i) for i in changed_indices]}")
        self.prev_enemy_memory = current_enemy_memory

    def step(self, action):
        # 处理动作
        self._take_action(action)

        # 获取新的状态
        state = self._get_obs()

        # 获取当前位置
        current_position = self.pyboy.memory[0xDBAE]

        # 检查位置是否不等于 0x3a
        if current_position != 0x3a:
            print("Position is not 0x3a, restarting the game!")
            self.pyboy.send_input(WindowEvent.STATE_LOAD)  # 按下 'x' 键
            self.pyboy.tick()  # 等待一帧以确保输入被处理
            self.pyboy.send_input(WindowEvent.STATE_LOAD)  # 释放 'x' 键
            state = self.reset()[0]  # 重置环境并获取初始状态

        # 检查生命值
        current_health = self.pyboy.memory[0xDB5A]  # 当前生命值
        if current_health == 0:
            # 生命值为零，按下 'x' 键重新开始
            print("Health is zero, restarting the game!")
            self.pyboy.send_input(WindowEvent.STATE_LOAD)  # 按下 'x' 键
            self.pyboy.tick()  # 等待一帧以确保输入被处理
            self.pyboy.send_input(WindowEvent.STATE_LOAD)  # 释放 'x' 键
            state = self.reset()[0]  # 重置环境并获取初始状态

        # 根据动作类型调整奖励, 这里是根据钥匙的位置来的，钥匙偏左，并且稍微偏上（训练过程总是向下走不知道为什么）
        if self.valid_actions[action] == WindowEvent.PRESS_ARROW_LEFT:
            additional_reward = 200  # 向左走加10点奖励
        elif self.valid_actions[action] == WindowEvent.PRESS_ARROW_RIGHT:
            additional_reward = -200  # 向右走减10点奖励
        elif self.valid_actions[action] == WindowEvent.PRESS_ARROW_UP:
            additional_reward = 10 # 向右走减10点奖励
        else:
            additional_reward = 0

            # 计算奖励
        reward = self._get_reward() + additional_reward

        # 检查是否结束
        done = self._is_done()

        # 检查是否结束
        terminated = self._is_done()
        truncated = False  # 如果有其他终止条件，可以在此设置 truncated 为 True

        info = {
            "TimeLimit.terminated": terminated,
            "TimeLimit.truncated": truncated
        }

        return state, reward, terminated, truncated, info

    def get_game_state_reward(self):
        # TODO:闲逛 - ， 获取自己的位置，

        # 自定义奖励函数，基于游戏状态的不同进行奖励计算
        current_health = self.pyboy.memory[0xDB5A]  # 获取当前生命值
        current_position = self.pyboy.memory[0xDBAE]  # 获取当前的位置（在这个例子中是0xDBAE地址）

        # 如果当前位置为0x3a，保持奖励不变，否则给与惩罚
        if current_position == 0x3a:
            position_reward = 100  # 位置为0x3a时不加奖励或惩罚
        else:
            position_reward = -100  # 位置不为0x3a时，进行惩罚
        

        # 修改：假设钥匙获取状态存储在特定内存地址，这里只是示例，需要根据实际情况修改
        key_status = self.pyboy.memory[0xDBD0] == 1
        if key_status and not self.key_obtained:
            key_reward = 100000  # 获得钥匙奖励1000
            self.key_obtained = True
            print("Congratulations! You have obtained the key!")  # 输出拿到钥匙的信息

        else:
            key_reward = 0

        # 生命值奖励，生命值降低给予负奖励
        health_reward = (current_health - self.last_health) * 10
        print(f"health: {health_reward}")
        self.last_health = current_health

        # 综合奖励
        self.cumulative_reward += key_reward + health_reward + position_reward

        return self.cumulative_reward

    def bit_count(self, value):
        return bin(value).count('1')

    def read_m(self, addr):
        return self.pyboy.memory[addr]

    def _get_reward(self):
        # 这里可以根据游戏状态计算奖励，例如使用 get_game_state_reward 方法
        return self.get_game_state_reward()

    def _is_done(self):
        # 这里可以根据游戏状态判断是否结束，例如生命值为 0 等
        # current_health = self.pyboy.memory[0xDB5A]
        return self.key_obtained == True

    def close(self):
        self.pyboy.stop()

config = {
    "session_path": r"D:\codes\codes_pycharm\da_chuang\sessions",  # 存储路径
    "save_video": True,  # 是否保存视频
    "headless": False,  # 是否无界面模式
    "action_freq": 1,  # 动作频率
    "max_steps": 10000,  # 最大步骤数
    "init_state": r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb.state"  # 初始化状态文件路径
}

from stable_baselines3 import PPO
# from stable_baselines3.common.envs import DummyVecEnv
import gym

# 使用环境
env = ZeldaEnv(rom_path=r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb", config=config)
# 将环境封装到一个 vectorized 环境中（Stable Baselines3 需要封装的环境）
# env = DummyVecEnv([lambda: env])  # 将环境包装成一个向量化环境，这对于并行训练非常有用

# 创建 PPO 模型
model = PPO("MultiInputPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=200000)  # 您可以根据需要调整 timesteps

# 保存模型

# 测试模型（根据训练的模型进行测试）
obs = env.reset()
for i in range(1000):  # 运行1000步
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()  # 如果您希望查看训练过程中的结果，可以启用渲染
    if dones:
        break

# 关闭环境
env.close()