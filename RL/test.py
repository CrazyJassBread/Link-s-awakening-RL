import gymnasium as gym
from stable_baselines3 import PPO
import imageio
import os
import time

from envs.env51_01 import Room51_Task1_Env as Zelda_Env

# ------------------ 配置 --------------------
MODEL_PATH = "RL/RL_model/ppo58_task2_final.zip"
SAVE_STATE = "game_state/Room58_task2.state"
GAME_FILE = "game_state/Link's awakening.gb"
MAX_FRAMES = 5000  # 最长多少帧后强制停止
# --------------------------------------------

# 创建环境（保持和训练时一致）
env = Zelda_Env(game_file=GAME_FILE, save_file=SAVE_STATE)
obs, _ = env.reset()

model = PPO.load(MODEL_PATH)
print("Model loaded successfully")
print(model.policy)

for i in range(MAX_FRAMES):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env.step(action)
    if i  % 50 == 0:
        print(f"Action: {action}")
        print(f"Step {_}: reward={reward}, done={done}, trunc={trunc}")
    if done or trunc:
        break
    time.sleep(0.1)  # 控制渲染速度
env.close()
