
from typing import Tuple, Dict, Literal

import numpy as np
import zmq
import subprocess
import psutil
from PIL import Image as im
import io
import struct
import time
import logging

little_endian = (struct.pack('@h', 1) == struct.pack('<h', 1))
if little_endian:
    endian_type = 'f'
    endian_order: Literal["little", "big"] = "little"
else:
    endian_type = '>f'
    endian_order: Literal["little", "big"] = "big"

unity_process: subprocess.Popen
unity_context: zmq.Context
unity_socket_action_pub: zmq.Socket
unity_socket_init_rep: zmq.Socket
unity_socket_obs_data_req: zmq.Socket
poller_req: zmq.Poller


def accurate_delay(delay: int):
    """
    Function to provide accurate time delay in milliseconds
    :param delay: Delay in milliseconds
    :return: Nothing
    """
    target_time = time.perf_counter() + delay/1000
    while time.perf_counter() < target_time:
        pass


def dec_to_base(num: int, base: int) -> np.ndarray:
    """
    Turns a base 10 int to a np.ndarray representing the int in base
    :param num: The int
    :param base: The base
    :return: An array representing the number in its new base
    """
    base_num = np.zeros(5).astype(int)
    i = 0
    while num > 0:
        dig = int(num % base)
        base_num[i] = dig
        i += 1
        num //= base
    base_num = np.flip(base_num)
    return base_num


def connect_sockets() -> bool:
    """
    Creates the ZMQ context and Sockets for the communication between the python process and the Unity exec one.
    :return: True if all initialises ok, False otherwise
    """
    global unity_context
    global unity_socket_action_pub
    global unity_socket_init_rep
    global unity_socket_obs_data_req
    global poller_req

    try:
        unity_context = zmq.Context()
        unity_socket_action_pub = unity_context.socket(zmq.PUB)
        unity_socket_action_pub.bind("tcp://*:12346")
        unity_socket_init_rep = unity_context.socket(zmq.REP)
        unity_socket_init_rep.bind("tcp://*:12344")
        unity_socket_obs_data_req = unity_context.socket(zmq.REQ)
        unity_socket_obs_data_req.bind("tcp://*:12345")
        poller_req = zmq.Poller()
        poller_req.register(unity_socket_obs_data_req, zmq.POLLIN)
    except Exception as e:
        print(e)
        return False

    return True


def start_unity_exe(unity_exe) -> bool:
    """
    Starts the Unity executable.
    :param unity_exe: The full path to the executable
    :return: True if the process starts, False if there is a problem
    """
    global unity_process
    logging.getLogger('PIL').setLevel(logging.WARNING)
    try:
        unity_process = subprocess.Popen(unity_exe)

        # The Unity process will be pinned to the same core as the Node. This overloads the core and Unity crashes.
        # The code below unpins the Unity process and lets the OS decide what to do.
        cores = []
        for cpu in range(psutil.cpu_count()):
            cores.append(cpu)
        psutil.Process(unity_process.pid).cpu_affinity(cores)
    except Exception as e:
        print(e)
        return False

    return True


def first_communication_with_unity(screen_res: Tuple[int, int] = (200, 200), translation_snap: float = 1,
                                   rotation_snap: int = 20, observation_type: str = 'Everything') -> bool:
    """
    Once the executable is running this function communicates with it for the first time, handshakes and passes the
    initialisation parameters to it.
    :param screen_res: The resolution of the game's screen (and thus the resolution of the pixels' observation)
    :param translation_snap: The amount of degrees the agent rotates per Rotation action
    :param rotation_snap: The amount of decimiters the agent moves per Move action
    :param observation_type: The type of the observations to be produced by Unity
    :return: The reward, pixels, features and the time in ms taken
    """
    try:
        # This will lock until Unity has sent a request
        unity_first_request = unity_socket_init_rep.recv_string()
        print("Unity executable's first handshake response is: {}".format(unity_first_request))

        unity_socket_init_rep.send_string('Python knows Unity is up.')
        accurate_delay(100)
        # Once the req rep handshake has happened then we can send commands to the Unity exe
        change_parameter('screen_res', '{}, {}'.format(screen_res[0], screen_res[1]))
        accurate_delay(100)
        change_parameter('move_snap', '{}'.format(translation_snap))
        accurate_delay(100)
        change_parameter('rotate_snap', '{}'.format(rotation_snap))
        accurate_delay(100)

        # Finally get the observation that has been prepared.
        #reward, pixels, features, ms_taken = get_observation(observation_type)

    except Exception as e:
        print(e)
        return False

    return True


def do_action(action: str):
    """
    Send an Action command to the environment.
    Possible actions: Move:Forwards, Move:Back, Rotate:CW, Rotate:CCW, Nothing:Nothing, LeftPaw:Extend, LeftPaw:Retract,
    RightPaw:Extend, RightPaw:Retract
    :param action: The string of the action
    :return: Nothing
    """
    unity_socket_action_pub.send_string('Action={}'.format(action))


def change_parameter(parameter_type, parameter_value):
    """
    Used to change certain environment parameters. The 'screen_res' parameter will only work once at the start of the
    Unity executable (called in the first_communication_with_unity() function). The executable will not return any
    observation pixels if the resolution parameter is not set!
    :param parameter_type: Possible Parameters: 'move_snap', 'rotate_snap', 'screen_res', 'reset'
    :param parameter_value: move_snap -> float, move_rotate -> int, screen_res -> int, int, reset -> bool (but not used)
    :return: Nothing
    """
    unity_socket_action_pub.send_string('Parameter={}:{}'.format(parameter_type, parameter_value))


def get_observation(observation_type):
    """
    This function is called right after a do_action. It will retrieve from the Unity game the prepared observations
    following the action. The observations can be either a np.ndarray of image data (channels last) if observation_type
    is 'Pixels', or a dict of features if observation_type is 'Features' or both if observation_type is 'Everything'.
    :param observation_type: string: Possible observation types: 'Pixels', 'Features', 'Everything'
    :return: reward, pixels, features, time_of_last_frame_in_ms
    """
    global unity_socket_obs_data_req
    global poller_req

    start_time = time.perf_counter()

    unity_socket_obs_data_req.send_string(observation_type)
    timeout = 100
    msgs = dict(poller_req.poll(timeout))

    pixels = None
    features = None
    reward = None

    if unity_socket_obs_data_req in msgs and msgs[unity_socket_obs_data_req] == zmq.POLLIN:

        if observation_type == 'Pixels':
            reward = get_reward()
            pixels = get_pixels()
        if observation_type == 'Features':
            features = get_features()
            reward = get_reward()
        if observation_type == 'Everything':
            features = get_features()
            reward = get_reward()
            pixels = get_pixels()

    else:
        unity_socket_obs_data_req.setsockopt(zmq.LINGER, 0)
        unity_socket_obs_data_req.close()
        poller_req.unregister(unity_socket_obs_data_req)
        unity_socket_obs_data_req = unity_context.socket(zmq.REQ)
        unity_socket_obs_data_req.bind("tcp://*:12345")
        poller_req = zmq.Poller()
        poller_req.register(unity_socket_obs_data_req, zmq.POLLIN)

    end_time = time.perf_counter()
    return reward, pixels, features, (end_time - start_time) * 1000


def get_pixels() -> np.ndarray:
    """
    Grabs the buffer data of the observation image as have arrived from Unity and decodes it into a np.ndarray
    :return: the np.ndarray of the image
    """
    data = unity_socket_obs_data_req.recv(flags=zmq.NOBLOCK)
    decoded = np.asarray(im.open(io.BytesIO(data)))
    decoded = np.flipud(decoded)
    return decoded


def get_features() -> dict:
    """
    Gets the dictionary of features from Unity given the game's communication protocol (see the Unity project for details)
    :return: the dict of the features
    """
    data = unity_socket_obs_data_req.recv(flags=zmq.NOBLOCK)
    number_of_features = int.from_bytes(data, endian_order)
    features_dict = {}
    for i in range(number_of_features):
        data = unity_socket_obs_data_req.recv(flags=zmq.NOBLOCK)
        name = data.decode('utf-8')
        data = unity_socket_obs_data_req.recv(flags=zmq.NOBLOCK)
        type_as_str = data.decode('utf-8')
        data = unity_socket_obs_data_req.recv(flags=zmq.NOBLOCK)
        num_of_values = int(data.decode('utf-8'))
        values = []
        for k in range(num_of_values):
            data = unity_socket_obs_data_req.recv(flags=zmq.NOBLOCK)
            values.append(type_from_byte(data, type_as_str))
        features_dict[name] = values
    return features_dict


def type_from_byte(byte, type_as_str) -> float | int | bool:
    """
    Turns the bytes of the features dict arriving from Unity into the correct python primitives
    :param byte: The arriving byte
    :param type_as_str: The string saying the type of the primitive
    :return: The final primitive
    """
    if type_as_str == 'float':
        return struct.unpack(endian_type, byte)[0]
    elif type_as_str == 'int':
        return int.from_bytes(byte, endian_order)
    elif type_as_str == 'bool':
        return struct.unpack('?', byte)[0]
    else:
        print("THE TYPE OF A FEATURE VALUE RECEIVED FROM UNITY IS NEITHER FLOAT, NOR INT NOR BOOL!!")


def get_reward() -> int:
    """
    Gets the reward of the current state from Unity
    :return: The reward as an int
    """
    reward = unity_socket_obs_data_req.recv(flags=zmq.NOBLOCK)
    reward = reward.decode('utf-8')
    reward = int(reward)
    return reward


def kill_unity():
    """
    Kills the Unity executable and closes any open sockets
    :return: Nothing
    """
    global unity_process
    global unity_socket_action_pub
    global unity_socket_init_rep
    global unity_socket_obs_data_req

    try:
        unity_process.kill()
        unity_socket_action_pub.close(linger=1)
        unity_socket_init_rep.close(linger=1)
        unity_socket_obs_data_req.close(linger=1)
    except:
        pass


def test_if_data_have_gone_through(reward, pixels, features, observation_type):
    if reward is None:
        return False
    if observation_type == 'Pixels' and pixels is None:
        return False
    if observation_type == 'Features' and (features is None or features == {}):
        return False
    if observation_type == 'Everything' and (pixels is None or (features is None or features == {})):
        return False
    return True


def connect(executable: str, observation_type: str, screen_res: Tuple[int, int],
            translation_snap: float, rotation_snap: int, use_unity_editor: bool = False) ->\
        Tuple[int, np.ndarray, Dict] | None:
    """
    Sets up the sockets, starts the Unity executable, connects to it and handshakes with it passing the initial
    parameters of the environment
    :param executable: The executable's full path
    :param observation_type: The type of the observations to be produced by Unity
    :param screen_res: The resolution of the game's screen (and thus the resolution of the pixels' observation)
    :param translation_snap: The amount of degrees the agent rotates per Rotation action
    :param rotation_snap: The amount of decimiters the agent moves per Move action
    :param use_unity_editor: If true the connect function doesn't start a separate executable but instead uses what is
    already running (e.g. the game running inside the Unity editor
    :return: The initial reward, feature dict and pixels
    """
    connect_sockets()
    if not use_unity_editor:
        start_unity_exe(executable)
    accurate_delay(3000)
    connection_state = first_communication_with_unity(screen_res=screen_res,
                                                      translation_snap=translation_snap,
                                                      rotation_snap=rotation_snap,
                                                      observation_type=observation_type)
    accurate_delay(1000)
    if connection_state:
        do_action('Nothing:Nothing')
        accurate_delay(5)
        reward, pixels, features, ms_taken = get_observation(observation_type)
        while not test_if_data_have_gone_through(reward, pixels, features, observation_type):
            do_action('Nothing:Nothing')
            accurate_delay(5)
            reward, pixels, features, ms_taken = get_observation(observation_type)
            accurate_delay(50)
    else:
        print('Cannot connect to Unity game.')
        return None

    return reward, pixels, features





