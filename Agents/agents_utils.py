
from collections import defaultdict
import numpy as np

class NumpyDefaultDict(defaultdict):
    def __init__(self, len_of_numpy_array):
        defaultdict.__init__(self, self.default_factory)
        self.len_of_numpy_array = len_of_numpy_array

    def default_factory(self):
         sub = np.zeros(self.len_of_numpy_array)
         return sub

    def savez_to_numpy(self, file_name):
        keys = np.array(list(self.keys()))
        values = np.array(list(self.values()))
        np.savez(file_name, keys=keys, values=values)

    @staticmethod
    def load_from_numpy(file_name: str):
        npz = np.load(file_name)
        len_of_numpy_array = len(npz['values'][0])
        temp_dict = NumpyDefaultDict(len_of_numpy_array)

        for i in np.arange(len(npz['keys'])):
            temp_dict[npz['keys'][i]] = npz['values'][i]

        return temp_dict