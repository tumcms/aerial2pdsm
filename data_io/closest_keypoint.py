import numpy as np
from survey.sfm_helpers import Keypoint


def ReadKeypointFile(path):
    kpoints = []
    with open(path, "r") as file:
        for line in file.readlines():
            if line[0] == "#":
                continue
            img_name, x, y = line.strip().split(" ")
            vec = np.array([x, y], dtype=np.float64)
            kpoints.append(Keypoint(img_name, vec))
    return kpoints
