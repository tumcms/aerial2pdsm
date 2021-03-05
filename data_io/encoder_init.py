import json
from pathlib import PosixPath
import numpy as np


class Aerial2PdsmEncoder(json.JSONEncoder):
    _registered = {}

    @classmethod
    def Register(cls, obj):
        cls._registered[obj.__name__] = obj

    def default(self, obj):
        if type(obj).__name__ in self.__class__._registered:
            return obj.to_dict()
        elif isinstance(obj, PosixPath):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)