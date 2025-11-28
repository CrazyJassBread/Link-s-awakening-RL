import os
import imageio
import numpy as np
from pyboy import PyBoy
from pynput import keyboard
from stable_baselines3.common.callbacks import BaseCallback

RECORD_FPS = 30          # 保存 GIF 时使用的 fps
MAX_FRAMES = 3000        # 防止无限增长
OUTPUT_GIF = "record/human_play/test.gif"

recording = True         # 初始即录制
running = True
frames = []
frame_count = 0

class SaveGifCallback(BaseCallback):
    def __init__(self, env_class, env_kwargs, save_path, save_interval, max_frames=1000, verbose=0):
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
        if self.verbose:
            print(f"Creating evaluation env and recording GIF to {gif_path}...")
        eval_env = self.env_class(**self.env_kwargs)

        frames = []
        try:
            obs, _ = eval_env.reset()
            for _ in range(self.max_frames):
                frame = eval_env.render(mode="rgb_array")
                if frame is not None:
                    frames.append(frame)
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, truncated, _ = eval_env.step(action)
                if done or truncated:
                    break
        except Exception as e:
            print(f"Error during GIF recording: {e}")
        finally:
            eval_env.close()

        if frames:
            imageio.mimsave(gif_path, frames, fps=30)
            if self.verbose:
                print(f"Saved GIF with {len(frames)} frames.")
        else:
            print("Warning: No frames were captured.")

def on_press(key):
    global last_save_state, running
    try:
        c = getattr(key, "char", None)
        if c:
            c = c.lower()
            if c == "q":
                running = False
                return False  # 停止监听器
    except Exception as e:
        print("键盘处理错误:", e)

def save_gif(path, frames_list):
    if not frames_list:
        print("No frames to save.")
        return
    print(f"Saving GIF to {path} with {len(frames_list)} frames...")
    imageio.mimsave(path, frames_list, fps=RECORD_FPS)

def main():
    global frame_count
    pyboy = PyBoy("game_state/Link's awakening.gb", window="null", sound_emulated=False)
    state_path = "game_state/Room58_task2.state"
    try:
        with open(state_path, "rb") as f:
            pyboy.load_state(f)
    except FileNotFoundError:
        print("start new game")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while running and pyboy.tick():
        if recording:
            frame_np = np.array(pyboy.screen.image)
            frames.append(frame_np)
            frame_count += 1
            if frame_count % 120 == 0:
                print(f"Recorded {frame_count} frames...")
            if frame_count >= MAX_FRAMES:
                print("Reached maximum frame count, stopping automatically.")
                break

    listener.stop()
    pyboy.stop()
    save_gif(OUTPUT_GIF, frames)
    print("Finished")

if __name__ == "__main__":
    main()
