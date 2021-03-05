import json
import sys
import numpy as np
import plyfile

from submodules.colmap_scripts.read_write_model import Point3D
from encoder_init import Aerial2PdsmEncoder


class Aerial2PdsmDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=Aerial2PdsmDecoder.default_hook, *args, **kwargs)

    @staticmethod
    def default_hook(dct):
        if "type" in dct:
            class_type = Aerial2PdsmEncoder._registered[dct["type"]]
            try:
                return class_type.FromDict(dct)
            except:
                e = sys.exc_info()[0]
                obj = class_type.__new__(class_type)
                obj.__dict__ = dct
                return obj
        return dct


def WritePly(pcloud: [Point3D], filename, precision="double"):

    _dtype = [("x", np.float64), ("y", np.float64), ("z", np.float64)]
    if precision == "float":
        _dtype = [("x", np.float32), ("y", np.float32), ("z", np.float32)]

    vertex = None
    try:
        vertex = np.array([tuple(i.xyz) for i in pcloud], dtype=_dtype)  # np.stack([mx, my, mz, m00.point3d.xyz])
    except:
        vertex = np.array([tuple(np.array(i)) for i in pcloud], dtype=_dtype)

    #if vertex.shape[0] > 1:
    #   vertex = vertex.squeeze(axis=1)
    edges = np.array([
        ([0, 3], 255, 0, 0),
        ([0, 3], 0, 255, 0),
        ([0, 3], 255, 0, 255)],
        dtype=[("index", "i4", (2,)),
               ("red", "u1"),
               ("green", "u1"),
               ("blue", "u1")])
    v = plyfile.PlyElement.describe(vertex, "vertices")
    e = plyfile.PlyElement.describe(edges, "edges")
    plyfile.PlyData([v], text=True).write(str(filename))
