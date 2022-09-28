

from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import os.path

from Agents.agents_utils import NumpyDefaultDict
from Environments.target_trap_env import TargetTrap
from Agents import q_learner_with_dynaq as ql

MAX_NUM_EPISODES = 1500
MAX_NUM_STEPS = 500
env_reset_options = {'target_trap': 'random', 'manipulandum': 'in'}
#env_reset_options = None

discretisation_bins = 36
alpha = 0.01
gamma = 1
starting_epsilon = 0.9
epsilon_decay = 0.000005
minimum_epsilon = 0.1

results_save_folder= r'/mnt/e/Code/Mine/Transfer_Learning/Transfer_Learning_ML_Pycharm_Project/Shaping_Modeling/' \
                    r'Random_vs_Transitions_Shaping/RL'
result_file = join(results_save_folder, 'q_learning_results_g1_a0p01_man_in.npz')
q_table_file = join(results_save_folder, 'q_table_g1_a0p01_man_in.npz')

#previous_q_table = NumpyDefaultDict.load_from_numpy(q_table_file)
previous_q_table = None

test_results = []
results_at_episode = []


def train(agent, env, save_results_file, save_Q_file):
    best_reward = -float('inf')

    total_reward = 0.0
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = env.reset(options=env_reset_options)
        #total_reward = 0.0
        step = 0
        while not done and step < MAX_NUM_STEPS:
            action = agent.get_action(obs)

            next_obs, reward, done, _, info = env.step(action)  # this is with the new step API of openAI gym
            agent.learn(obs, action, reward, next_obs)
            #print(list(env.decode(obs)), action, list(env.decode(next_obs)), done)
            obs = next_obs
            #env.render()
            total_reward += reward
            step += 1

        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode,
                                     total_reward, best_reward, agent.epsilon))

        if episode % 2 == 0:
            policy = compute_policy()
            num_of_epis = 100
            try:
                test_results.append(test(agent, env, policy, env_reset_options, episodes=num_of_epis) / num_of_epis)
                results_at_episode.append(episode)
                np.savez(save_results_file, test_results=np.array(test_results), results_at_episode=np.array(results_at_episode))

                agent.Q.savez_to_numpy(save_Q_file)
            except:
                pass

def compute_policy():
    pol = {}
    for key in agent.Q.keys():
        pol[key] = agent.Q[key].argmax()
    return pol

def single_test(agent, env, policy, env_reset_options, max_steps=500):
    done = False
    obs = env.reset(options=env_reset_options)
    tot_rew = 0.0
    step = 0
    while step < max_steps and not done:
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, _, info = env.step(action)  # deals with the new step API of openAI gym
        obs = next_obs
        if reward > 0:
            tot_rew += reward
        step += 1
    return tot_rew


def test(agent, env, policy, env_reset_options, max_steps=100, episodes=1000):
    total_reward = 0
    for _ in range(episodes):
        total_reward += single_test(agent, env, policy, env_reset_options, max_steps)
    return total_reward


env = TargetTrap(rot_speed_deg_per_sec=20)
env.reset(options=env_reset_options)

agent = ql.Q_Learner(env=env, discretisation_bins=discretisation_bins, alpha=alpha, gamma=gamma,
                     starting_epsilon=starting_epsilon, epsilon_decay=epsilon_decay, minimum_epsilon=minimum_epsilon,
                     Q=previous_q_table)
train(agent, env, result_file, q_table_file)

learned_policy = compute_policy()

q_learning_results = np.load(result_file)
plt.plot(q_learning_results['results_at_episode'], q_learning_results['test_results'])
plt.plot(q_learning_results['test_results'])



f = plt.Figure()
a = plt.subplot(111)
labels = ['Full / g=0.9', 'Full / g=1', 'Man In / g=0.9', 'Man In / g=1']
for i, s in enumerate(['g0p9_a0p01', 'g1_a0p01', 'g0p9_a0p01_man_in', 'g1_a0p01_man_in']):
    res_file = join(results_save_folder, 'q_learning_results_{}.npz'.format(s))
    lr = np.load(res_file)
    a.plot(lr['results_at_episode'], lr['test_results'], label=labels[i])
a.legend()
a.set_xlabel('Trial')
a.set_ylabel('Percent correct in 100 sessions test')

Q = NumpyDefaultDict.load_from_numpy(q_table_file)

episodes=1000
print(test(agent, env, learned_policy, env_reset_options, episodes=episodes) / episodes)

"""

import gym
env = gym.make("Taxi-v3", new_step_api=True)
env.reset()

discretisation_bins = 20
alpha = 0.2
gamma =1
starting_epsilon = 0.9
epsilon_decay = 0.000001
minimum_epsilon = 0.1

results_save_folder= r'/mnt/e/Code/Mine/Transfer_Learning/Transfer_Learning_ML_Pycharm_Project/Tests'

agent = ql.Q_Learner(env=env, discretisation_bins=discretisation_bins, alpha=alpha, gamma=gamma,
                     starting_epsilon=starting_epsilon, epsilon_decay=epsilon_decay, minimum_epsilon=minimum_epsilon)

learned_policy = train(agent, env, results_save_folder)
"""
