
import gym
import numpy as np
from functools import reduce


class DiscreteBox:
    """
    Discretises an openAI gym Box space into a set number of bins to allow Box observation spaces to be used with
    tabular Q-learner agents
    """
    def __init__(self, box_space: gym.spaces.box.Box, number_of_bins_per_dimension: int):
        assert type(box_space) == gym.spaces.box.Box

        self.box_space = box_space
        self.number_of_bins_per_dimension = number_of_bins_per_dimension

        self.bin_sizes = (box_space.high - box_space.low) / self.number_of_bins_per_dimension
        self.shape = box_space.shape

        self.number_of_bins = np.power(self.number_of_bins_per_dimension,
                                       reduce(lambda x, y: x * y, self.box_space.shape))

    def get_discrete_sample(self, sample):
        '''
        Given a sample from Box this returns the discrete version with the same dimensionality. So if box.shape = (2,3)
        this function returns a (2,3) array where its element is the discrete bin for the corresponding dimension
        :param sample: a Box.sample()
        :return: The discretised version of the sample
        '''
        return np.floor((sample - self.box_space.low) / self.bin_sizes)

    def get_discrete_state_bin(self, sample):
        '''
        Given a sample from Box this return a single integer that corresponds to the unique bin of the flattened tensor
        of the discretised sample. So if sample.shape = (2,3) then there will be a total of
        [self.number_of_bins_per_dimension ^ (2+3)] unique bins and this function will return the one that corresponds
        to the sample given.
        :param sample: a Box.sample()
        :return: The unique state bin that corresponds to the sample
        '''
        discrete_sample = self.get_discrete_sample(sample)
        discrete_sample_flattened = discrete_sample.flatten()
        result = 0
        for i, s in enumerate(discrete_sample_flattened):
            result = result + s * np.power(self.number_of_bins_per_dimension, i)
        return result

    def sample(self):

        sample = self.box_space.sample()
        return self.get_discrete_state_bin(sample)


class DiscreteWrapper(gym.spaces.discrete.Discrete):
    def __init__(self, discrete_space: gym.spaces.discrete.Discrete):
        assert type(discrete_space) == gym.spaces.discrete.Discrete
        super().__init__(discrete_space.n)

        self.discrete_space = discrete_space

    def get_discrete_sample(self, sample):
        return sample

    def get_discrete_state_bin(self, sample):
        return sample


class FlatMultiDiscrete:
    """
    Class that takes a MultiDiscrete space and flattens it (used in tabular q-learners)
    """
    def __init__(self, multi_discrete_space: gym.spaces.MultiDiscrete):
        assert type(multi_discrete_space) == gym.spaces.multi_discrete.MultiDiscrete
        self.multi_discrete_space = multi_discrete_space
        self.number_of_bins_per_dimension = self.multi_discrete_space.nvec

        self.number_of_bins = self._get_number_of_total_bins()

        self.flat_discrete = gym.spaces.Discrete(self.number_of_bins)

    def _get_number_of_total_bins(self):
        """
        :return: The number of total states (bins) of the MultiDiscrete space
        """
        result = reduce(lambda x, y: x * y, self.number_of_bins_per_dimension)
        try:
            result.shape[0] > 1
            while True:
                result = reduce(lambda x, y: x * y, result)
                try:
                    result.shape[0] > 1
                except:
                    break
        except:
            pass
        return result

    def get_discrete_state_bin(self, sample):
        """
        Gets a sample and turns it into the integer that represents the state bin of the flattened space
        :param sample: a MultiDiscrete sample
        :return: An int representing the state bin (counter)
        """
        discrete_sample_flattened = sample.flatten()
        nvec_flat = self.multi_discrete_space.nvec.flatten()
        result = 0
        for i, s in enumerate(discrete_sample_flattened):
            if i == 0:
                result = s
            else:
                result = result + s * reduce(lambda x, y: x*y, nvec_flat[:i])
        return result

    def sample(self):
        """
        Generate a random sample and returns the flat state bin
        :return: The int that is the state bin of the randomly generated sample
        """
        s = self.multi_discrete_space.sample()
        return self.get_discrete_state_bin(s)


class FlatDictSpace:
    def __init__(self, dict_space, number_of_bins_per_dimension: int = None):
        """
        Constructor of the FlatDictSpace
        :param dict_space: The gym.spaces.dict.Dict space
        :param number_of_bins_per_dimension: If the Dict space has any Box spaces that need to be discretised then the number of
        bins needs to be specified. This parameter will be used to define te number of bins all of the Boxes' dimensions
        will be discretised into
        """
        assert type(dict_space) == gym.spaces.dict.Dict

        self.dict_space = dict_space
        self.number_of_bins_per_dimension = number_of_bins_per_dimension
        self.number_of_bins = 0

        self.discrete_dict = {}

        self._calculate_number_of_bins()

    def _calculate_number_of_bins(self):
        for key in self.dict_space:
            if type(self.dict_space[key]) == gym.spaces.box.Box:
                self.discrete_dict[key] = DiscreteBox(self.dict_space[key], self.number_of_bins_per_dimension)
                if self.number_of_bins == 0:
                    self.number_of_bins = self.discrete_dict[key].number_of_bins
                else:
                    self.number_of_bins = self.number_of_bins * self.discrete_dict[key].number_of_bins
            elif type(self.dict_space[key]) == gym.spaces.multi_discrete.MultiDiscrete:
                self.discrete_dict[key] = FlatMultiDiscrete(self.dict_space[key])
                if self.number_of_bins == 0:
                    self.number_of_bins = self.discrete_dict[key].number_of_bins
                else:
                    self.number_of_bins = self.number_of_bins * self.discrete_dict[key].number_of_bins
            elif type(self.dict_space[key]) == gym.spaces.discrete.Discrete:
                self.discrete_dict[key] = self.dict_space[key]
                if self.number_of_bins == 0:
                    self.number_of_bins = self.dict_space[key].n
                else:
                    self.number_of_bins = self.number_of_bins * self.dict_space[key].n

    def get_discrete_sample(self, sample):
        discrete_sample = dict.fromkeys(self.discrete_dict.keys())
        for key in discrete_sample:
            if type(self.discrete_dict[key]) == DiscreteBox:
                discrete_sample[key] = self.discrete_dict[key].get_discrete_state_bin(sample[key])
            elif type(self.dict_space[key]) == gym.spaces.multi_discrete.MultiDiscrete:
                discrete_sample[key] = self.discrete_dict[key].get_discrete_state_bin(sample[key])
            elif type(self.dict_space[key]) == gym.spaces.discrete.Discrete:
                discrete_sample[key] = sample[key]

        return discrete_sample

    def get_discrete_state_bin(self, sample):
        discrete_sample_flattened = list(self.get_discrete_sample(sample).values())
        nvec_flat = []
        for key in self.discrete_dict.keys():
            if type(self.discrete_dict[key]) == gym.spaces.discrete.Discrete:
                nvec_flat.append(self.discrete_dict[key].n)
            else:
                nvec_flat.append(self.discrete_dict[key].number_of_bins)
        result = 0
        for i, s in enumerate(discrete_sample_flattened):
            if i == 0:
                result = s
            else:
                result = result + s * reduce(lambda x, y: x * y, nvec_flat[:i])
        return result

    def sample(self):
        sample = self.dict_space.sample()
        return self.get_discrete_state_bin(sample)


