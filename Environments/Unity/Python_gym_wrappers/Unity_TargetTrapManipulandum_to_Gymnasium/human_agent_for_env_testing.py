
import os

import numpy as np

from Environments.Unity.Python_gym_wrappers.Unity_TargetTrapManipulandum_to_Gymnasium.gymnasium_ttm_wrapper import \
    TargetTrapManipulandum as TTM_Env
import dearpygui.dearpygui as dpg
from stable_baselines3.common.env_checker import check_env

# Callbacks ----------
reward = None
pixels = None
features = None
time_of_frame = None
total_reward = 0

#path for the laptop
path_to_unity_exe = os.path.join(r'E:\\', 'Code', 'Mine', 'Transfer_Learning', 'transfer_learning_ML', 'Environments',
                                 'Unity', 'Target_Trap_Manipulandum', 'Builds')
# path for the desktop
path_to_unity_exe = os.path.join(r'E:\\', 'Software', 'Develop', 'Source', 'Repos', 'RL', 'transfer_learning_ML',
                                 'Environments', 'Unity', 'Target_Trap_Manipulandum', 'Builds')

game_exe = 'TTM_ExploreCorners'
observation_type = 'Everything'
action_space_type = 'Full'

ttm_env = TTM_Env(path_to_unity_builds=path_to_unity_exe, game_executable=game_exe, observation_type=observation_type,
                  action_space_type=action_space_type, screen_res=(100, 100), move_snap=0.4, rotate_snap=20)


# Create the initial image
texture_data = []
for i in range(0, 100 * 100):
    texture_data.append(100 / 255)
    texture_data.append(100 / 255)
    texture_data.append(100 / 255)
    texture_data.append(255 / 255)


def step(sender, app_data, user_data):
    global reward
    global pixels
    global features
    global time_of_frame
    global total_reward

    action = user_data

    obs, reward, done, _, info = ttm_env.step(action)
    if type(obs) == dict:
        pixels = obs['Pixels']
        features = obs['Features']
    elif type(obs) == np.ndarray:
        if len(obs.shape) > 1:
            pixels = obs
        else:
            features = obs

    total_reward += reward
    dpg.set_value('Reward', reward)
    dpg.set_value('Time', time_of_frame)
    dpg.set_value('Total Reward', total_reward)

    if pixels is not None:
        new_texture_data = pixels.flatten('K')
        new_texture_data = new_texture_data/255
        new_texture_data = new_texture_data.tolist()
        dpg.set_value("pixels_obs_tag", new_texture_data)

    if features is not None:
        dpg.set_value('Features', '{}'.format(features).replace('], ', "]\n"))
    else:
        dpg.set_value('Features', 'None\n\n\n\n\n\n\n')
# ----------------------

# THE GUI --------------
dpg.create_context()

with dpg.texture_registry(show=True):
    dpg.add_dynamic_texture(width=100, height=100, default_value=texture_data, tag="pixels_obs_tag")


dpg.create_viewport(title='Rat RL', width=520, height=750)

with dpg.window(label="TTM GUI", width=520, height=750):
    with dpg.group(horizontal=True, horizontal_spacing=20):
        dpg.add_button(label="Connect", callback=ttm_env.reset, indent=50, width=150)
        dpg.add_button(label="Disconnect", callback=ttm_env.close, indent=0, width=150)

    dpg.add_spacer(height=30)

    dpg.add_button(label="Forwards", callback=step, user_data=1, indent=150, width=150)
    with dpg.group(horizontal=True, horizontal_spacing=20):
        dpg.add_button(label="CCW", callback=step, user_data=4, indent=50, width=150)
        dpg.add_button(label="CW", callback=step, user_data=3, indent=0, width=150)
    dpg.add_button(label="Back", callback=step, user_data=2, indent=150, width=150)
    dpg.add_button(label="Nothing", callback=step, user_data=0, indent=150, width=150)
    with dpg.group(horizontal=True, horizontal_spacing=20):
        dpg.add_button(label="LeftPaw Extend", callback=step, user_data=5, indent=50, width=150)
        dpg.add_button(label="RightPaw Extend", callback=step, user_data=7, indent=0, width=150)
    with dpg.group(horizontal=True, horizontal_spacing=20):
        dpg.add_button(label="LeftPaw Retrieve", callback=step, user_data=6, indent=50, width=150)
        dpg.add_button(label="RightPaw Retrieve", callback=step, user_data=8, indent=0, width=150)

    dpg.add_spacer(height=50)
    dpg.add_text(label="ImageLabel", default_value='Pixels Observation', indent=100)
    with dpg.group(horizontal=True, horizontal_spacing=50):
        dpg.add_spacer(width=70)
        dpg.add_image("pixels_obs_tag")

    dpg.add_spacer(height=20)

    dpg.add_text(label="Features Label", default_value='Features Dictionary', indent=100)
    dpg.add_text(label="Features", default_value='None\n\n\n\n\n\n\n', tag='Features', indent=80)

    dpg.add_spacer(height=20)

    with dpg.group(horizontal=True, horizontal_spacing=50):
        dpg.add_text(label="Reward", tag='Reward', default_value='0', indent=100)
        dpg.add_text(label="RewardLabel", default_value='Reward', indent=150)
    with dpg.group(horizontal=True, horizontal_spacing=50):
        dpg.add_text(label="Time of Frame / ms", default_value='0', tag='Time', indent=100)
        dpg.add_text(label="Time of Frame / ms abel", default_value='Time of Frame / ms', indent=150)
    with dpg.group(horizontal=True, horizontal_spacing=50):
        dpg.add_text(label="Total Reward", default_value='0', tag='Total Reward', indent=100)
        dpg.add_text(label="Total Reward Label", default_value='Total Reward', indent=150)


dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

