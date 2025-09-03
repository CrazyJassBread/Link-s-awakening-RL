from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import Zelda_Env, game_file, save_file

# 加载游戏环境
save_file = "RL/game_state/Link's awakening.gb.state"
game_file = "RL/game_state/Link's awakening.gb"

def make_env():
    return Zelda_Env(game_file=game_file, save_file=save_file)

env = DummyVecEnv([make_env])

# 加载模型
model = PPO.load("RL/RL_model/ppo_zelda_final", env=env)

# 测试模型
obs = env.reset()
total_reward = 0.0
n_steps = 1000  # 测试步数

for _ in range(n_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    total_reward += rewards[0]  # rewards是数组，取第一个环境
    if dones[0]:                # dones也是数组
        obs = env.reset()

print(f"Total reward over {n_steps} steps: {total_reward}")
env.close()
