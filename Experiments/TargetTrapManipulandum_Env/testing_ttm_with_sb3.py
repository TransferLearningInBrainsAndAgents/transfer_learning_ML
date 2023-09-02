
import os
from Environments.Unity.Python_gym_wrappers.Unity_TargetTrapManipulandum_to_Gymnasium.gymnasium_ttm_wrapper \
    import TargetTrapManipulandum_UnityWrapper_Env
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO


path_to_unity_exe = os.path.join(r'E:\\', 'Code', 'Mine', 'Transfer_Learning', 'transfer_learning_ML', 'Environments',
                                 'Unity', 'Target_Trap_Manipulandum', 'Builds')
tensorboard_log = os.path.join(r'E:\\', 'Code', 'Mine', 'Transfer_Learning', 'transfer_learning_ML', 'Experiments',
                               'TargetTrapManipulandum_Env', 'tensorboard_logs', 'Recurrent_PPO')

game_exe = 'TTM_ExploreCorners'
observation_type = 'Features'
screen_res = (100, 100)
move_snap = 0.1
rotate_snap = 10

ttm_env = TargetTrapManipulandum_UnityWrapper_Env(path_to_unity_builds=path_to_unity_exe, game_executable=game_exe,
                                                  observation_type=observation_type, screen_res=screen_res, move_snap=move_snap,
                                                  rotate_snap=rotate_snap)


# Define and Train the agent
model = RecurrentPPO("MlpLstmPolicy", ttm_env, tensorboard_log=tensorboard_log,
                     learning_rate=1e-6).learn(total_timesteps=100000)

model_save_file = r'E:\Code\Mine\Transfer_Learning\transfer_learning_ML\Experiments\TargetTrapManipulandum_Env\tensorboard_logs\RecurrentPPO_5\recurrentPPO_onTTM_ExploreCorners.zip'
model.save(model_save_file)
