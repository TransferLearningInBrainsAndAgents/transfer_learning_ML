

from datetime import datetime
import os
import pickle
from Environments.Unity.Python_gym_wrappers.Unity_TargetTrapManipulandum_to_Gymnasium.gymnasium_ttm_wrapper \
    import TargetTrapManipulandum_UnityWrapper_Env
import torch
from rnn_sac.sac.sac import SAC

torch.set_default_device('cuda:0')

base_folder = os.path.join(r'E:\\', 'Software', 'Develop', 'Source', 'Repos', 'RL')
path_to_unity_exe = os.path.join(base_folder, 'transfer_learning_ML', 'Environments', 'Unity',
                                 'Target_Trap_Manipulandum', 'Builds')
base_tensorboard_log = os.path.join(base_folder, 'transfer_learning_ML', 'Experiments', 'TargetTrapManipulandum_Env',
                                    'tensorboard_logs', 'Discreet_SAC')

game_exe = 'TTM_ExploreCorners'
observation_type = 'Features'
action_space_type = 'Full'  # 'Simple' or 'Full'
screen_res = (100, 100)
move_snap = 0.1
rotate_snap = 10
save_observations = True

ttm_env = TargetTrapManipulandum_UnityWrapper_Env(path_to_unity_builds=path_to_unity_exe, game_executable=game_exe,
                                                  observation_type=observation_type, action_space_type=action_space_type,
                                                  screen_res=screen_res, move_snap=move_snap, rotate_snap=rotate_snap,
                                                  save_observations=save_observations)

# Define and Train the agent
logger_kwargs = {'output_dir': os.path.join(base_tensorboard_log,
                                            str(datetime.now()).rpartition(':')[0].replace('-', '_').replace(' ', '-').
                                            replace(':', '_'))}
epochs = 100
number_of_trajectories = 10
max_ep_len = 10000
lr = 1e-4
gamma_lr = 0.95
epochs_to_update_lr = 20
gamma = 0.99
seed = 43
update_every = 10
save_every_n_update = 2  # That means the model will be saved every  save_every_n_update * update_every trajectories
polyak = 0.95
batch_size = 10
hidden_size = 256
start_steps = 200
exploration_sampling = False
clip_ratio = 0.95
use_alpha_annealing = True
entropy_target_mult = 0.95
model_file_to_load = None
#model_file_to_load = os.path.join(base_tensorboard_log, '2023_09_19-16_41', 'pyt_save', 'actor_critic_model_6_9.pt')

model = SAC(env=ttm_env, logger_kwargs=logger_kwargs, seed=seed, max_ep_len=max_ep_len,
            save_every_n_update=save_every_n_update, gamma=gamma, lr=lr, gamma_lr=gamma_lr,
            epochs_to_update_lr=epochs_to_update_lr, polyak=polyak, epochs=epochs, batch_size=batch_size,
            hidden_size=hidden_size, start_steps=start_steps, update_every=update_every,
            exploration_sampling=exploration_sampling, clip_ratio=clip_ratio,
            number_of_trajectories=number_of_trajectories, use_alpha_annealing=use_alpha_annealing,
            entropy_target_mult=entropy_target_mult,
            model_file_to_load=model_file_to_load)

model.train_agent(ttm_env)

with open(os.path.join(model.logger.output_dir, 'all_observations.pkl'), 'wb') as f:
    pickle.dump(ttm_env.save_observations, f)

for i in range(10):
    print('---- {} -----'.format(i))
    observations, rewards = model.test_agent(test_env=ttm_env, num_test_episodes=4, random_init=1000, greedy_ratio=0.2)


# Plot path
import matplotlib.path

with open(os.path.join(model.logger.output_dir, 'all_observations.pkl'), 'rb') as f:
    obs = pickle.load(f)



obs = np.array(ttm_env.save_observations)[:, :2]
colours = np.array([[i/obs.shape[0], 0, 0, 1] for i in range(obs.shape[0])])

from matplotlib.collections import LineCollection
lc = LineCollection(segments=[obs], colors=colours)
fig, ax = plt.subplots()
ax.set_xlim(obs[:, 0].min(), obs[:, 0].max())
ax.set_ylim(obs[:, 1].min(), obs[:, 1].max())
ax.add_collection(lc)

plt.plot(obs[:, 0], obs[:, 1])
