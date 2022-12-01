
import gym
from gym import spaces
import numpy as np


class Arena(gym.Env):
    metadata = {}

    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y
        