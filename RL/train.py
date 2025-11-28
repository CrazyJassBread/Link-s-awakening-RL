import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn
import torch
import numpy as np
import imageio
import os

from envs.base_env import BaseEnv
from envs.env58_02 import Room58_Task2_Env as Zelda_Env
from PPO.model import CustomResNet, CustomACPolicy, CustomPPO, TQDMProgressBar

TOTAL_STEPS = 3000000
SAVE_INTERVAL = 100000
GIF_SAVE_PATH = "record/PPO/ppo58_task2_gifs/"

save_state = "game_state/Room58_task2.state"
game_file = "game_state/Link's awakening.gb"

env_kwargs = {
    "game_file": game_file,
    "save_file": save_state, 
}
os.makedirs(GIF_SAVE_PATH, exist_ok=True)

env = Zelda_Env(game_file=game_file, save_file=save_state)
env.disable_render = True
env = Monitor(env)

class SaveGifCallback(BaseCallback):
    def __init__(self, env_class, env_kwargs, save_path, save_interval, max_frames=1000, verbose=0):
        """
        :param env_class: 环境的类对象 (例如 Zelda_Env)
        :param env_kwargs: 初始化环境需要的参数字典
        :param save_path: GIF 保存路径
        :param save_interval: 保存间隔
        :param max_frames: 限制录制的最大帧数，防止无限循环
        """
        super(SaveGifCallback, self).__init__(verbose)
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.save_path = save_path
        self.save_interval = save_interval
        self.max_frames = max_frames
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_interval == 0:
            gif_filename = f"step_{self.num_timesteps}.gif"
            gif_path = os.path.join(self.save_path, gif_filename)
            self._save_gif(gif_path)
        return True
    
    def _save_gif(self, gif_path):
        if self.verbose > 0:
            print(f"Creating evaluation env and recording GIF to {gif_path}...")
        eval_env = self.env_class(**self.env_kwargs)
        frames = []
        try:
            obs, _ = eval_env.reset()
            for _ in range(self.max_frames):
                # 获取图像
                frame_np = np.array(eval_env.pyboy.screen.image)
                if frame_np is not None:
                    frames.append(frame_np)

                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, truncated, _ = eval_env.step(action)
                if done or truncated:
                    break
        except Exception as e:
            print(f"Error during GIF recording: {e}")
        finally:
            eval_env.close()

        if len(frames) > 0:
            imageio.mimsave(gif_path, frames, fps=30)
            if self.verbose > 0:
                print(f"Saved GIF with {len(frames)} frames.")
        else:
            print("Warning: No frames were captured.")

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
    normalize_advantage=False,
    tensorboard_log="./log/Room58/ppo_tensorboard/"
)

gif_callback = SaveGifCallback(
    env_class=Zelda_Env,
    env_kwargs=env_kwargs,
    save_path=GIF_SAVE_PATH,
    save_interval=SAVE_INTERVAL
)

model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True, callback=gif_callback)
model.save("RL/RL_model/test/ppo58_task2_final")
env.close()