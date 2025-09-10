import numpy as np
import random
from pyboy import PyBoy
import gymnasium as gym
from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from skimage.transform import downscale_local_mean

TOTAL_STEPS = 1000000
MAX_STEPS = 4000
