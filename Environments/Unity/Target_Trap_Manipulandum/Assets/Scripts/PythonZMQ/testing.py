

import numpy as np
import zmq
import subprocess
import os
import time
import cv2

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

def start_unity_exe(path_to_unity_exe=path_to_unity_exe):
    global unity_process
    try:
        unity_process = subprocess.Popen(path_to_unity_exe)
        print(unity_process)
    except Exception as e:
        print(e)
        return False

    print(first_communication_with_unity())

    return True


def first_communication_with_unity():
    try:
        # That will lock until Unity has send a request
        unity_first_request = unity_socket_init_rep.recv_string()
        unity_socket_init_rep.send_string('Python knows Unity is up.')
        time.sleep(0.1)
        # Once the req rep handshake has happened then we can send commands to the Unity exe

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


def get_observation(observation_type):
    """
    Possible observation types: 'Pixels', 'Parameters', Everything
    :param observation_type:
    :return:
    """
    global unity_socket_obs_data_req
    global poller_req

    unity_socket_obs_data_req.send_string(observation_type)

    timeout = 10
    msgs = dict(poller_req.poll(timeout))

    if unity_socket_obs_data_req in msgs and msgs[unity_socket_obs_data_req] == zmq.POLLIN:
        data = unity_socket_obs_data_req.recv(flags=zmq.NOBLOCK)
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
        decoded = np.flipud(decoded)
        return decoded
    else:
        unity_socket_obs_data_req.setsockopt(zmq.LINGER, 0)
        unity_socket_obs_data_req.close()
        poller_req.unregister(unity_socket_obs_data_req)
        unity_socket_obs_data_req = unity_context.socket(zmq.REQ)
        unity_socket_obs_data_req.bind("tcp://*:12345")
        poller_req = zmq.Poller()
        poller_req.register(unity_socket_obs_data_req, zmq.POLLIN)
        return None


def kill():
    unity_process.kill()
    unity_socket_action_pub.close(linger=1)
    unity_socket_init_rep.close(linger=1)
    unity_socket_obs_data_req.close(linger=1)


start_unity_exe(path_to_unity_exe=path_to_unity_exe)


frame_num = 0
avg_frame_times = []
while(True):
    start_frame = frame_num
    start_time = time.perf_counter()
    for k in range(36):
        do_action('Rotate', 'CW')
        obs = get_observation('')
        frame_num += 1
        for i in range(10):
            do_action('Move', 'Back')
            obs = get_observation('')
            frame_num += 1
        for i in range(10):
            do_action('Move', 'Forwards')
            obs = get_observation('')
            frame_num += 1
    num_of_frames = frame_num - start_frame
    d_time = time.perf_counter() - start_time
    avg_frame_times.append(1000 * d_time / num_of_frames)
    print(frame_num, num_of_frames, d_time)


