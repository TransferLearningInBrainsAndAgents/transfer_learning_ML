
import gym
from gym import spaces
import numpy as np


class TargetTrap(gym.Env):
    metadata = {}

    def __init__(self, rot_speed_deg_per_sec=20):
        self.rot_speed_deg_per_sec = rot_speed_deg_per_sec
        self.updates_per_sec = 3  # How many actions can the agent perform in a second
        self.degrees_per_step = int(360 / (self.updates_per_sec * self.rot_speed_deg_per_sec))

        self._target_trap_state = 0
        self._manipulandum_angle = 0

        self.keys = {0: 'target_trap_orientation', 1: 'manipulandum_angle'}
        # The target_trap_orientation space denotes whether the target points at 12 or 9 o'clock (states 1 and 2) or
        # the target and trap are invisible (state 0)
        # The manipulandum_angle denotes whether the manipulandum is invisible (0) or visible (>0) and at a specific
        # angle in respect to the vertical.
        self.observation_space = spaces.Dict(spaces={self.keys[0]: spaces.Discrete(3),
                                                     self.keys[1]: spaces.Box(0, 360, dtype=int)})

        # The action space can be: Do nothing (0), Press left button (1) or Press right button (2)
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
        return {self.keys[0]: self._target_trap_state, self.keys[1]: self._manipulandum_angle}

    def linear_to_circular(self, number):
        number = number % 361
        if number == 0:
            number = 1
        return number

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)

        if options is None:
            in_or_out = 'in' if np.random.rand() > 0.5 else 'out'
            options = {'target_trap': 'random', 'manipulandum': in_or_out}

        # The options dict carries info on how the target and trap should be initialised ('up' means the target points
        # up and the trap right and 'right' means the target points right and the trap up).
        assert options['target_trap'] == 'random' or options['target_trap'] == 'up', options['target_trap'] == 'right'

        # The options dict carries info on whether the manipulandum will start in an angle between 10 and 80 (in) or
        # 100 and 350 (out)
        assert options['manipulandum'] == 'in' or options['manipulandum'] == 'out'

        if options['target_trap'] == 'random':
            self._target_trap_state = np.random.randint(low=1, high=3, size=1)[0]
        elif options['target_trap'] == 'up':  # Target is up, trap is right
            self._target_trap_state = 1
        elif options['target_trap'] == 'right':  # Target is right, trap is up
            self._target_trap_state = 2

        if options['manipulandum'] == 'in':
            self._manipulandum_angle = np.random.randint(10, 80, 1)[0]
        elif options['manipulandum'] == 'out':
            self._manipulandum_angle = np.random.randint(100, 350, 1)[0]

        observation = self._get_obs()
        
        return observation

    def step(self, action):
        done = False
        reward = 0

        if action == 1:
            self._manipulandum_angle += self.degrees_per_step
            self._manipulandum_angle = self.linear_to_circular(self._manipulandum_angle)

        elif action == 2:
            self._manipulandum_angle -= self.degrees_per_step
            self._manipulandum_angle = self.linear_to_circular(self._manipulandum_angle)

        if self._manipulandum_angle < self.degrees_per_step:  # Just crossed the 0 degrees
            done = True
            if self._target_trap_state == 1:
                reward = +1
            elif self._target_trap_state == 2:
                reward = -1

        if self._manipulandum_angle > 90 and self._manipulandum_angle < 90 + self.degrees_per_step:  # Just crossed the 90 degrees
            done = True
            if self._target_trap_state == 1:
                reward = -1
            elif self._target_trap_state == 2:
                reward = +1

        obs = self._get_obs()
        info = {}

        return obs, reward, done, None, info

    def render(self, mode=None):
        pass

    def close(self):
        pass