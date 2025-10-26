import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from gymnasium import spaces

from gymnasium.spaces import Box, Dict

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        residual = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual
    

class CustomResNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 1024):
        super().__init__(observation_space, features_dim)

        # 取出真实的图像空间
        if isinstance(observation_space, Dict):
            image_space = observation_space["game_area"]
            shape = image_space.shape
        elif isinstance(observation_space, spaces.Box):
            shape = observation_space.shape
        else:
            raise ValueError(f"Unsupported observation space type: {type(observation_space)}")

        if len(shape) != 3:
            raise ValueError(f"Expect 3D image obs, got shape={shape}")

        # 既支持 CHW 也支持 HWC
        if shape[0] in (1, 3, 4):   # CHW
            c, h, w = shape
        elif shape[2] in (1, 3, 4): # HWC
            h, w, c = shape
        else:
            # 兜底：按 HWC 处理
            h, w, c = shape

        self.in_channels = c

        self.conv_net = nn.Sequential(
            nn.Conv2d(c, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResNetBlock(64),
            ResNetBlock(64),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResNetBlock(64),
            ResNetBlock(64),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResNetBlock(128),
            ResNetBlock(128),
        )

        with torch.no_grad():
            sample_input = torch.zeros(1, c, h, w)  # 按 CHW 构造样本
            conv_output = self.conv_net(sample_input)
            n_flatten = conv_output.reshape(1, -1).shape[1]

        self.dense_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 在使用 VecTransposeImage 后会把输入变为 CHW。
        # 若仍是 HWC，则仅在需要时转为 CHW。
        if observations.dim() == 4 and observations.shape[1] != self.in_channels and observations.shape[-1] == self.in_channels:
            observations = observations.permute(0, 3, 1, 2).contiguous()

        features = self.conv_net(observations)
        return self.dense_net(features)


class CustomACPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Modify final layer initialization
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.orthogonal_(self.value_net.weight, gain=0.1)
        nn.init.constant_(self.action_net.bias, 0)
        nn.init.constant_(self.value_net.bias, 0)


class EWMARolloutBuffer(RolloutBuffer):
    def __init__(self, *args, ewma_decay=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewma_decay = ewma_decay
        self.ewma_mean = 0.0
        self.ewma_var = 1.0
        self.ewma_count = 0

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        # Compute returns and advantages using parent method
        super().compute_returns_and_advantage(last_values, dones)
        
        # Update EWMA statistics using returns from the current rollout
        returns = self.returns.flatten()
        batch_mean = np.mean(returns)
        batch_var = np.var(returns)
        
        if self.ewma_count == 0:
            self.ewma_mean = batch_mean
            self.ewma_var = batch_var
        else:
            # Apply EWMA update
            self.ewma_mean = self.ewma_decay * self.ewma_mean + (1 - self.ewma_decay) * batch_mean
            self.ewma_var = self.ewma_decay * self.ewma_var + (1 - self.ewma_decay) * batch_var
        
        self.ewma_count += 1
        
        # Normalize returns with EWMA stats
        self.returns = (self.returns - self.ewma_mean) / (np.sqrt(self.ewma_var) + 1e-8)


class CustomPPO(PPO):
    def _setup_model(self) -> None:
        super()._setup_model()  # Initialize default components
        # Replace the buffer with the custom EWMA buffer
        self.rollout_buffer = EWMARolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            ewma_decay=0.99  # As per your settings
        )


    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)

        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

class TQDMProgressBar(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.ewma_mean = 0.0
        self.ewma_var = 1.0

    def _on_training_start(self):
        # Initialize progress bar
        self.pbar = tqdm(total=self.total_timesteps, dynamic_ncols=True)
        self.pbar.set_description("Training PPO")

    def _on_step(self):
        # Update progress bar
        self.pbar.update(self.training_env.num_envs)  # For parallel environments
        return True

    def _on_rollout_end(self):
        # Add custom metrics (modify with your actual metrics)
        stats = {
            "ep_rew_mean": np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]),
            "ewma_mean": self.model.rollout_buffer.ewma_mean,
            "ewma_var": self.model.rollout_buffer.ewma_var,
            "entropy": self.model.logger.name_to_value["train/entropy_loss"]
        }
        self.pbar.set_postfix(stats)

    def _on_training_end(self):
        # Close progress bar
        self.pbar.close()
