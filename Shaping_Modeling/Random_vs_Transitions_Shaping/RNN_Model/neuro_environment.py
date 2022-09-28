

import numpy as np

import neurogym as ngym
from neurogym import spaces


class RotateToMatchWithTrap_V0(ngym.TrialEnv):
    """
    Args:
        speed: float, the rotational speed of the manipulandum in degrees per second

    """
    metadata = {
    }

    def __init__(self, dt=100, rewards=None, speed=30):
        super().__init__(dt=dt)

        self.rewards = {'correct': +1., 'fail': -0.1}
        if rewards:
            self.rewards.update(rewards)

        self.speed = speed
        self.degrees_per_step = 360 / self.speed

        self.observation_space = spaces.Dict(spaces={'target_trap_orientation':
                                                         spaces.Discrete(2, name='target_trap_orientation'),
                                                     'manipulandum_angle':
                                                         spaces.Box(0, 359, dtype=float, name='manipulandum_angle')})
        name = {'do_noting': 0, 'left_lever': 1, 'right_lever': 2}
        self.action_space = spaces.Discrete(3, name=name)

    def _new_trial(self, **kwargs):
        """
        self._new_trial() is called internally to generate a next trial.
        Typically, you need to
            set trial: a dictionary of trial information
            run self.add_period():
                will add time periods to the trial
                accesible through dict self.start_t and self.end_t
            run self.add_ob():
                will add observation to np array self.ob
            run self.set_groundtruth():
                will set groundtruth to np array self.gt
        Returns:
            trial: dictionary of trial information
        """



    def _step(self, action):
        pass