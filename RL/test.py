import gymnasium as gym
from stable_baselines3 import PPO
import imageio
import os
import time

from PPO.envs.env51_01 import Room51_Task1_Env as Zelda_Env

# ------------------ 配置 --------------------
MODEL_PATH = "RL/RL_model/test/ppo51_task1_final.zip"
GIF_PATH = "RL/gifs/test/ppo51_task1_test.gif"
SAVE_STATE = "RL/game_state/Room51_task1.state"
GAME_FILE = "RL/game_state/Link's awakening.gb"
MAX_FRAMES = 5000  # 最长多少帧后强制停止
# --------------------------------------------

# 创建环境（保持和训练时一致）
env = Zelda_Env(game_file=GAME_FILE, save_file=SAVE_STATE)
env.disable_render = False   # 测试时需要渲染
obs, _ = env.reset()

# 加载训练好的模型
model = PPO.load(MODEL_PATH)
print("Model loaded successfully")
print(model.policy)
# 用于存放 GIF 帧
frames = []

# 开始测试
for i in range(MAX_FRAMES):
    frame = env.render(mode="rgb_array")
    frames.append(frame)

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env.step(action)
    if i  % 50 == 0:
        print(f"Action: {action}")
        print(f"Step {_}: reward={reward}, done={done}, trunc={trunc}")
    if done or trunc:
        break
    time.sleep(0.02)  # 控制渲染速度

env.close()

# 保存 GIF
os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
print(f"Total frames: {len(frames)}")
imageio.mimsave(GIF_PATH, frames, fps=30)
print(f"✅ GIF saved at: {GIF_PATH}")
