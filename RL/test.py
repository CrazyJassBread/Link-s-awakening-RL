import gymnasium as gym
from stable_baselines3 import PPO
import time

from envs.env58_02 import Room58_Task2_Env as Zelda_Env

MODEL_PATH = "RL/RL_model/ppo58_task2_final.zip"
SAVE_STATE = "game_state/Room58_task2.state"
GAME_FILE = "game_state/Link's awakening.gb"
MAX_FRAMES = 5000

env = Zelda_Env(game_file=GAME_FILE, save_file=SAVE_STATE, render_mode="human", goal_room=58)
obs, _ = env.reset()

model = PPO.load(MODEL_PATH)
print("Model loaded successfully")
print(model.policy)

steps = 0
for steps in range(MAX_FRAMES):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env.step(action)
    if steps  % 5 == 0:
        print(f"Action: {action}")
        print(f"Step {steps}: reward={reward}, done={done}, trunc={trunc}")
    if done or trunc:
        break
    time.sleep(0.1)

print("Episode finished after {} steps".format(steps + 1))
env.close()
