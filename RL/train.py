import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from env import Zelda_Env, game_file, save_file

import os
"""
checkpoint_dir = "./checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,  # 每训练 10k 步保存一次模型
    save_path=checkpoint_dir,
    name_prefix="ppo_zelda"
)
"""

TOTAL_STEPS = 1000000

save_file = "RL\game_state\Room_58.state"
game_file = "RL\game_state\Link's awakening.gb"

env = Zelda_Env(game_file=game_file, save_file=save_file)
env = Monitor(env)

"""
# 检查环境是否封装完好
from gymnasium.utils.env_checker import check_env
try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
"""

model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=512,
    n_epochs=3,
    gamma=0.95,
    gae_lambda=0.65,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="./ppo_zelda_tensorboard/"
)


model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
model.save("RL\RL_model\ppo_58_final")
env.close()