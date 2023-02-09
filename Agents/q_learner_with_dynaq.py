
import gym
import numpy as np
import random
from Agents.agents_utils import NumpyDefaultDict
from Environments.gym.env_utils import FlatDictSpace, DiscreteBox, DiscreteWrapper


class Q_Learner(object):
    def __init__(self, env, discretisation_bins, alpha=0.1, gamma=0.9, starting_epsilon=0.9,
                 epsilon_decay=1e-4, minimum_epsilon=0.1, Q=None):

        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = starting_epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon

        if type(env.observation_space) == gym.spaces.box.Box:
            self.discrete_observation_space = DiscreteBox(box_space=env.observation_space,
                                                          number_of_bins_per_dimension=discretisation_bins)
        elif type(env.observation_space) == gym.spaces.dict.Dict:
            self.discrete_observation_space = FlatDictSpace(dict_space=env.observation_space,
                                                            number_of_bins_per_dimension=discretisation_bins)
        elif type(env.observation_space) == gym.spaces.discrete.Discrete:
            self.discrete_observation_space = DiscreteWrapper(discrete_space=env.observation_space)

        if Q is None:
            self.create_q_table(self.env)
        else:
            self.Q = Q

        self.obs_bin_to_name = {}

    def create_q_table(self, env):
        action_size = env.action_space.n
        self.Q = NumpyDefaultDict(len_of_numpy_array=action_size)

    def discretize(self, obs):
        return self.discrete_observation_space.get_discrete_state_bin(obs)

    def get_action(self, obs):
        discretized_obs = self.discretize(obs)
        self.obs_bin_to_name[discretized_obs] = obs
        # Epsilon-Greedy action selection
        if self.epsilon > self.minimum_epsilon:
            self.epsilon -= self.epsilon_decay * self.epsilon

        if np.random.random() > self.epsilon:
            action = np.argmax(self.Q[discretized_obs])
        else:  # Choose a random action
            action = random.choice(np.arange(self.env.action_space.n))

        return action

    def learn(self, obs, action, reward, next_obs):
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        Qsa_next = np.max(self.Q[discretized_next_obs]) if discretized_next_obs is not None else 0
        td_target = reward + self.gamma * np.max(Qsa_next)
        td_error = td_target - self.Q[discretized_obs][action]
        self.Q[discretized_obs][action] = self.Q[discretized_obs][action] + self.alpha * td_error



