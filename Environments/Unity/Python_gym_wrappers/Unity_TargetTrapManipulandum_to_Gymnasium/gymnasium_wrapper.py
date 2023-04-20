
from typing import Tuple, Dict
import os
import gymnasium as gym
import numpy as np

from Environments.Unity.Python_gym_wrappers.Unity_TargetTrapManipulandum_to_Gymnasium import \
    unity_communication_protocol as ucp


class TargetTrapManipulandum(gym.Env):
    """
    The class that creates a gymnasium environment to communicate with the Unity executable that actually runs
    the environment.

    :param path_to_unity_builds: (str) The full path to the  folder where the Unity project's builds are (the executables
            that are the different environments this Unity project has made)
    :param game_executable: (str) The name of the executable to run. Currently, the possibilities are:
            TTM_ButtonsNoPoke, TTM_ButtonsWithPoke, TTM_ExploreCorners, TTM_FindReward, TTM_WaitForReward. Each one of
            these is a different environment but with the same action space and very similar observation spaces
    :param observation_type: (str) The possible observation types the Unity game will return. Unity creates as
            observations both the image captured from the point of view of the agent in the scene and a dictionary
            that carries a set of features like the agents position in the arena, the agent's rotation, etc. The
            observation_type defines which of these observations should be returned
            Can be: 'Pixels', 'Features', 'Everything'. 'Everything' means both the features and the pixels.
    :param screen_res: tuple(int, int) The screen resolution of the game thus of the image that the observations returns
            The 'Pixels' observation from Unity is then of shape (screen_res[0],
    :param move_snap: (float) How much does the agent move with every 'Move' command (in meters, so the 0.1 default is
            10 cm)
    :param rotate_snap: (int) How much the agent will rotate with every 'Rotate' command (in degrees, so the default 10
            is 10 degrees)
    """
    def __init__(self, path_to_unity_builds: str, game_executable: str, observation_type: str = 'Features',
                 screen_res: Tuple[int, int] = (100, 100), move_snap: float = 0.1, rotate_snap: int = 10):

        self.observation_type = observation_type
        self.screen_res = screen_res
        self.translation_snap = move_snap
        self.rotation_snap = rotate_snap
        self.size_of_arena = 8  # in decimeters, same units as the move_snap
        self.game = game_executable

        self.path_to_unity_exe = os.path.join(path_to_unity_builds,  game_executable + '.exe')

        self.observation_space = self.generate_observation_space()
        self.action_dict = {0: 'Nothing:Nothing', 1: 'Move:Forwards', 2: 'Move:Back', 3: 'Rotate:CW', 4: 'Rotate:CCW',
                            5: 'LeftPaw:Extend', 6: 'LeftPaw:Retrieve', 7: 'RightPaw:Extend', 8: 'RightPaw:Retrieve'}
        self.action_space = gym.spaces.Discrete(len(self.action_dict))

        self.info = {'Game': self.game, 'Observation Type': self.observation_type, 'Move Snap': self.translation_snap,
                     'Rotate Snap': self.rotation_snap, 'Arena size': self.size_of_arena, 'Time of State Update': 0}

    def generate_observation_space(self) -> gym.spaces.MultiDiscrete | gym.spaces.Box | gym.spaces.Dict:
        if self.observation_type == 'Features':
            return self.generate_feature_multidiscrete_obs_space()
        if self.observation_type == 'Pixels':
            return gym.spaces.Box(low=0, high=255, shape=(4, self.screen_res[0], self.screen_res[1]))
        if self.observation_type == 'Everything':
            return gym.spaces.Dict({'Pixels': gym.spaces.Box(low=0, high=255, shape=(4, self.screen_res[0],
                                                                                     self.screen_res[1])),
                                    'Features': self.generate_feature_multidiscrete_obs_space()})

    def generate_feature_multidiscrete_obs_space(self) -> gym.spaces.MultiDiscrete:
        """
        Creates a gym.spaces.MultiDiscrete observational space given the number of possible states the
        i) Rat X position
        ii) Rat Y position
        iii) Rat Rotation
        iv) Left Paw State
        v) Right Paw State
        vi) Target and Trap State
        vii) Manipulandum Rotation
        can be in
        vi and vii exist only for the games (environments) that use the Target Trap Manipulandum system
        (TTM_ButtonsNoPoke, TTM_ButtonsWithPoke and TTM_WaitForReward)

        The Unity game also sends the positions of the two buttons as observation features, but these are not used
        in the MultiDiscrete observation space

        :return: the gym.spaces.MultiDiscreet space
        """
        number_of_bins_per_dimension = [int(np.ceil(self.size_of_arena / self.translation_snap)),  # Rat X positions
                                        int(np.ceil(self.size_of_arena / self.translation_snap)),  # Rat Y positions
                                        int(np.ceil(360 / self.rotation_snap)),  # Rat Rotations
                                        2,  # Left Paw Extended State
                                        2]  # Right Paw Extended State
        if 'FindReward' not in self.game and 'ExploreCorners' not in self.game:
            number_of_bins_per_dimension.append(2)  # Positions of Target and Trap
                                                    # (Vertical or Horizontal Target with perpendicular Trap)
            number_of_bins_per_dimension.append(360)  # Angles of Manipulandum

        return gym.spaces.MultiDiscrete(number_of_bins_per_dimension)

    def generate_multidiscrete_sample_from_unity_features_dict(self, features: Dict) -> np.ndarray:
        """
        Takes in the features dictionary returned by the Unity game and transforms it into a np.ndarray as if it was
        a sample from the MultiDiscrete gym space the environment uses to wrap the feature observations in
        :param features: The Unity returned Dict
        :return: The features transformed into a gym.spaces.MultiDiscrete sample
        """

        sample = []
        # The +0.5*self.size_of_arena is because 0, 0 is in the middle of the arena and the Rat Position ranges
        # between -0.5 * self.size_of_arena and +0.5 * self.size_of_arena
        sample.append(int(np.ceil(features['Rat Position'][0] + 0.5 * self.size_of_arena / self.translation_snap)))
        sample.append(int(np.ceil(features['Rat Position'][1] + 0.5 * self.size_of_arena / self.translation_snap)))
        sample.append(int(np.ceil(features['Rat Rotation'][0]) / self.rotation_snap))
        sample.append(int(features['Left Paw Extended'][0]))
        sample.append(int(features['Right Paw Extended'][0]))
        if 'FindReward' not in self.game and 'ExploreCorners' not in self.game:
            sample.append(int(features['Target Trap State'][0]))
            sample.append(int(np.ceil(features['Manipulandum Angle'])))

        return np.array(sample)

    def generate_observation(self, pixels, features) -> np.ndarray | Dict:
        """
        Given the features and pixels having arrived from Unity this function bundles everything together in an
        observation that respects the observation space of the environment
        :param pixels: The np.ndarray of the image
        :param features: The dict of the features
        :return: an observation
        """
        if pixels is not None:
            pixels = pixels.transpose((2, 0, 1))
        if features is not None:
            features = self.generate_multidiscrete_sample_from_unity_features_dict(features)

        if self.observation_type == 'Pixels':
            obs = pixels
        elif self.observation_type == 'Features':
            obs = features
        elif self.observation_type == 'Everything':
            obs = {'Pixels': pixels, 'Features': features}

        return obs

    def step(self, action: int) -> Tuple[np.ndarray | Dict, int, bool, bool, Dict]:
        """
        Does a step in the environment
        :param action: The action of the step
        :return: The observation, reward, terminated (always False), truncated (always False) and info of the step
        """
        action_str = self.action_dict[action]
        ucp.do_action(action_str)

        reward, pixels, features, ms_taken = ucp.get_observation(self.observation_type)

        obs = self.generate_observation(pixels, features)
        terminated = False
        truncated = False

        self.info['Time of State Update'] = ms_taken

        return obs, reward, terminated, truncated, self.info

    def reset(self, seed=None, return_info=False, options=None):
        """
        Resetting the environment entails deleting any opened Unity executable and starting a new one

        :param seed: No randomness is required
        :param return_info: no return_info
        :param options: no options
        :return:
        """
        ucp.kill_unity()
        reward, pixels, features = ucp.connect(self.path_to_unity_exe, self.observation_type, self.screen_res,
                                               self.translation_snap, self.rotation_snap)
        obs = self.generate_observation(pixels, features)
        return obs, self.info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        """
        Kills the Unity exec and closes down all ZMQ sockets
        :return: Nothing
        """
        ucp.kill_unity()