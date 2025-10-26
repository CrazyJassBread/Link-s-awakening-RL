import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn
import torch

from PPO.envs.base_env import BaseEnv
from PPO.envs.env51_01 import Room51_Task1_Env as Zelda_Env
from PPO.model import CustomResNet, CustomACPolicy, CustomPPO, TQDMProgressBar

import imageio
import os



TOTAL_STEPS = 3000000
SAVE_INTERVAL = 300000
GIF_SAVE_PATH = "RL/gifs/test/ppo59_task1_gifs/"

save_state = "RL/game_state/Room59_task1.state"
game_file = "RL/game_state/Link's awakening.gb"

os.makedirs(GIF_SAVE_PATH, exist_ok=True)

env = Zelda_Env(game_file=game_file, save_file=save_state)
env.disable_render = True
env = Monitor(env)

# Device selection: read LA_DEVICE env (e.g. 'cuda:0' or 'cpu'), otherwise prefer CUDA if available
import torch
la_device = os.getenv("LA_DEVICE")
if la_device:
    device = la_device
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class SaveGifCallback(BaseCallback):
    def __init__(self, save_path, save_interval, verbose=0):
        super(SaveGifCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_interval = save_interval

    def _on_step(self) -> bool:
        # 每隔 save_interval 步保存一次 GIF
        if self.num_timesteps % self.save_interval == 0:
            gif_path = os.path.join(self.save_path, f"step_{self.num_timesteps}.gif")
            self._save_gif(gif_path)
        return True
    
    def _save_gif(self, gif_path):
        frames = []
        env = self.training_env.envs[0].unwrapped  # 获取当前环境
        original_disable_render = getattr(env, "disable_render", False)
        env.disable_render = False

        obs, _ = env.reset()
        for _ in range(1000):  # 渲染 200 帧
            frames.append(env.render(mode="rgb_array"))
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            if done:
                break

        env.disable_render = original_disable_render

        env.close()
        # 保存为 GIF
        imageio.mimsave(gif_path, frames, fps=30)
        if self.verbose > 0:
            print(f"Saved GIF to {gif_path}")
"""
# 检查环境是否封装完好
from gymnasium.utils.env_checker import check_env
try:
    check_env(env.unwrapped)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
"""

'''
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
    tensorboard_log="./test/ppo_zelda_tensorboard/"
)
'''
policy_kwargs = {
    "features_extractor_class": CustomResNet,
    "features_extractor_kwargs": {"features_dim": 1024},
    "activation_fn": nn.ReLU,
    "net_arch": [],
    "optimizer_class": torch.optim.Adam,
    "optimizer_kwargs": {"eps": 1e-5}
}

model = CustomPPO(
    CustomACPolicy,
    env,
    device=device,
    policy_kwargs=policy_kwargs,
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
    normalize_advantage=False
)

gif_callback = SaveGifCallback(save_path=GIF_SAVE_PATH, save_interval=SAVE_INTERVAL)

model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True, callback=gif_callback)

model.save("RL/RL_model/test/ppo59_task1_final")
env.close()
