

from datetime import datetime
import os
from Environments.Unity.Python_gym_wrappers.Unity_TargetTrapManipulandum_to_Gymnasium.gymnasium_ttm_wrapper \
    import TargetTrapManipulandum
import torch
torch.set_default_device('cuda:0')

base_folder = os.path.join(r'E:\\', 'Software', 'Develop', 'Source', 'Repos', 'RL')
path_to_unity_exe = os.path.join(base_folder, 'transfer_learning_ML', 'Environments', 'Unity',
                                 'Target_Trap_Manipulandum', 'Builds')
base_tensorboard_log = os.path.join(base_folder, 'transfer_learning_ML', 'Experiments', 'TargetTrapManipulandum_Env',
                                    'tensorboard_logs', 'Discreet_SAC')

game_exe = 'TTM_ExploreCorners'
observation_type = 'Features'
action_space_type = 'Simple'  # 'Simple' or 'Full'
screen_res = (100, 100)
move_snap = 0.1
rotate_snap = 10

ttm_env = TargetTrapManipulandum(path_to_unity_builds=path_to_unity_exe, game_executable=game_exe,
                                 observation_type=observation_type, action_space_type = action_space_type,
                                 screen_res=screen_res, move_snap=move_snap, rotate_snap=rotate_snap)

# Define and Train the agent
from rnn_sac.sac.sac import SAC


model_file_to_load = os.path.join(base_tensorboard_log, '2023_05_22-15_38', 'pyt_save', 'actor_critic_model.pt')

env = ttm_env
logger_kwargs = {'output_dir': os.path.join(base_tensorboard_log,
                                            str(datetime.now()).rpartition(':')[0].replace('-', '_').replace(' ', '-').
                                            replace(':', '_'))}
number_of_trajectories = 1000
max_ep_len = 5000
lr = 1e-4
gamma = 0.99
seed = 42
save_freq = 1
ac_kwargs = dict()
polyak = 0.95
epochs = 1
batch_size = 100
hidden_size = 256
start_steps = 500
update_after = 1000
update_every = 2
exploration_sampling = False
clip_ratio = 1.0
use_alpha_annealing = True
model_file_to_load = None
model_file_to_load = os.path.join(base_tensorboard_log, '2023_06_12-09_13', 'pyt_save', 'actor_critic_model.pt')

model = SAC(env=env, logger_kwargs=logger_kwargs, seed=seed, max_ep_len=max_ep_len, save_freq=save_freq,
            gamma=gamma, lr=lr, ac_kwargs=ac_kwargs, polyak=polyak, epochs=epochs,
            batch_size=batch_size, hidden_size=hidden_size,
            start_steps=start_steps, update_after=update_after, update_every=update_every,
            exploration_sampling=exploration_sampling, clip_ratio=clip_ratio,
            number_of_trajectories=number_of_trajectories, use_alpha_annealing=use_alpha_annealing,
            model_file_to_load=model_file_to_load)

model.train_agent(ttm_env)


model.test_agent(test_env=ttm_env, num_test_episodes=1, random_init=400)