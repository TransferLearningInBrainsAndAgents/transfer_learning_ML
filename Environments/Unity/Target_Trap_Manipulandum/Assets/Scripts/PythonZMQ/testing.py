

import numpy as np
import zmq
import subprocess
import os
import time
import cv2
import matplotlib.pyplot as plt
import dearpygui.dearpygui as dpg


unity_context = zmq.Context()
unity_socket_action_pub = unity_context.socket(zmq.PUB)
unity_socket_action_pub.bind("tcp://*:12346")
unity_socket_init_rep = unity_context.socket(zmq.REP)
unity_socket_init_rep.bind("tcp://*:12344")
unity_socket_obs_data_req = unity_context.socket(zmq.REQ)
unity_socket_obs_data_req.bind("tcp://*:12345")
poller_req = zmq.Poller()
poller_req.register(unity_socket_obs_data_req, zmq.POLLIN)


path_to_unity_exe = os.path.join(r'E:\Code\Mine\Transfer_Learning\transfer_learning_ML\Environments\Unity',
                              'Target_Trap_Manipulandum', 'Builds', 'Target_Trap_Manipulandum.exe')


def accurate_delay(delay):
    """
    Function to provide accurate time delay in millisecond
    :param delay: Delay in milliseconds
    :return: Nothing
    """
    target_time = time.perf_counter() + delay/1000
    while time.perf_counter() < target_time:
        pass


def start_unity_exe(path_to_unity_exe=path_to_unity_exe, screen_res=(200, 200)):
    global unity_process
    try:
        unity_process = subprocess.Popen(path_to_unity_exe)
        print(unity_process)
    except Exception as e:
        print(e)
        return False

    print(first_communication_with_unity(screen_res))

    return True


def first_communication_with_unity(screen_res=(200, 200)):
    try:
        # That will lock until Unity has send a request
        unity_first_request = unity_socket_init_rep.recv_string()
        unity_socket_init_rep.send_string('Python knows Unity is up.')
        time.sleep(0.1)
        # Once the req rep handshake has happened then we can send commands to the Unity exe
        change_parameter('screen_res', '{}, {}'.format(screen_res[0], screen_res[1]))
    except Exception as e:
        print(e)
        return False

    return unity_first_request


def do_action(action_type, action_value):
    """
    Send an Action command to the environment.
    Possible actions: Move:Forwards, Move:Back, Rotate:CW, Rotate:CCW
    :param action_type: 'Move' or 'Rotate'
    :param action_value: 'Forwards' or 'Back' for 'Move', 'CW' or 'CCW' for Rotate
    :return: Nothing
    """
    unity_socket_action_pub.send_string('Action={}:{}'.format(action_type, action_value))


def change_parameter(parameter_type, parameter_value):
    """
    Used to change certain environment parameters. The 'screen_res' parameter will only work once at the start of the
    Unity executable (called in the first_communication_with_unity() function). The executable will not return any
    observation pixels if the resolution parameter is not set!
    :param parameter_type: Possible Parameters: 'move_snap', 'rotate_snap', 'screen_res'
    :param parameter_value: move_snap -> float, move_rotate -> int, screen_res -> int, int
    :return: Nothing
    """
    unity_socket_action_pub.send_string('Parameter={}:{}'.format(parameter_type, parameter_value))


def get_observation(observation_type):
    """
    Possible observation types: 'Pixels', 'Parameters', Everything
    :param observation_type:
    :return:
    """
    global unity_socket_obs_data_req
    global poller_req

    start_time = time.perf_counter()
    unity_socket_obs_data_req.send_string(observation_type)
    timeout = 100
    msgs = dict(poller_req.poll(timeout))

    if unity_socket_obs_data_req in msgs and msgs[unity_socket_obs_data_req] == zmq.POLLIN:
        data = unity_socket_obs_data_req.recv(flags=zmq.NOBLOCK)
        reward = unity_socket_obs_data_req.recv(flags=zmq.NOBLOCK)
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
        decoded = np.flipud(decoded)
        reward = reward.decode('utf-8')
        end_time = time.perf_counter()
        return reward, decoded, (end_time - start_time) * 1000
    else:
        unity_socket_obs_data_req.setsockopt(zmq.LINGER, 0)
        unity_socket_obs_data_req.close()
        poller_req.unregister(unity_socket_obs_data_req)
        unity_socket_obs_data_req = unity_context.socket(zmq.REQ)
        unity_socket_obs_data_req.bind("tcp://*:12345")
        poller_req = zmq.Poller()
        poller_req.register(unity_socket_obs_data_req, zmq.POLLIN)
        end_time = time.perf_counter()
        return None, None, (end_time - start_time) * 1000


def kill():
    unity_process.kill()
    unity_socket_action_pub.close(linger=1)
    unity_socket_init_rep.close(linger=1)
    unity_socket_obs_data_req.close(linger=1)


def connect():
    global reward
    global obs
    global time_of_frame
    start_unity_exe(path_to_unity_exe=path_to_unity_exe, screen_res=(100, 100))
    accurate_delay(2000)
    reward, obs, ms_taken = get_observation('Pixels')


# Callbacks ----------
reward = None
obs = None
time_of_frame = None


def step(sender, app_data, user_data):
    global reward
    global obs
    global time_of_frame

    action_type = user_data[0]
    action_value = user_data[1]

    do_action(action_type, action_value)
    reward, obs, time_of_frame = get_observation('Pixels')
    dpg.set_value('Reward', reward)
    dpg.set_value('Time', time_of_frame)
# ----------------------


dpg.create_context()
dpg.create_viewport(title='Rat RL', width=500, height=300)

with dpg.window(label="TTM GUI", width=500, height=300):
    with dpg.group(horizontal=True, horizontal_spacing=20):
        dpg.add_button(label="Connect", callback=connect, indent=50, width=100)
        dpg.add_button(label="Disconnect", callback=kill, indent=0, width=100)

    dpg.add_spacer(height=20)

    dpg.add_button(label="Forwards", callback=step, user_data=('Move', 'Forwards'), indent=100, width=100)
    with dpg.group(horizontal=True, horizontal_spacing=20):
        dpg.add_button(label="CCW", callback=step, user_data=('Rotate', 'CCW'), indent=50, width=100)
        dpg.add_button(label="CW", callback=step, user_data=('Rotate', 'CW'), indent=0, width=100)
    dpg.add_button(label="Back", callback=step, user_data=('Move', 'Back'), indent=100, width=100)
    dpg.add_button(label="Nothing", callback=step, user_data=('Nothing', 'Nothing'), indent=100, width=100)
    with dpg.group(horizontal=True, horizontal_spacing=20):
        dpg.add_button(label="LeftPaw Extend", callback=step, user_data=('LeftPaw', 'Extend'), indent=50, width=100)
        dpg.add_button(label="RightPaw Extend", callback=step, user_data=('RightPaw', 'Extend'), indent=0, width=100)
    with dpg.group(horizontal=True, horizontal_spacing=20):
        dpg.add_button(label="LeftPaw Retrieve", callback=step, user_data=('LeftPaw', 'Retrieve'), indent=50, width=100)
        dpg.add_button(label="RightPaw Retrieve", callback=step, user_data=('RightPaw', 'Retrieve'), indent=0, width=100)

    dpg.add_spacer(height=20)

    dpg.add_text(label="Reward", tag='Reward', indent=10, show_label=True)
    dpg.add_text(label="Time of Frame / ms", tag='Time', indent=0, show_label=True)


dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()




'''
do_action('Move', 'Forward')
reward, obs, ms_taken = get_observation('Pixels')
plt.clf()
plt.imshow(obs)


# Measuring timings
obs, ms_taken = get_observation('')
frame_num = 0
avg_frame_times = []
n = 0
while(n < 10):
    start_frame = frame_num
    start_time = time.perf_counter()
    for k in range(36):
        do_action('Rotate', 'CW')
        reward, obs, ms_taken = get_observation('')
        frame_num += 1
        for i in range(10):
            do_action('Move', 'Back')
            reward, obs = get_observation('')
            frame_num += 1
        for i in range(10):
            do_action('Move', 'Forwards')
            reward, obs = get_observation('')
            frame_num += 1
    num_of_frames = frame_num - start_frame
    d_time = time.perf_counter() - start_time
    avg_frame_times.append(1000 * d_time / num_of_frames)
    print(frame_num, num_of_frames, d_time)
    n += 1
'''
