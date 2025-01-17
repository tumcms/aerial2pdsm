import plyfile

from config import SparseModel, project_path
from colmap_scripts.read_model import Image
import numpy as np

project = SparseModel("/home/felix/pointclouds/_working/2019_11_07_Muenchen_26_10_2018",
                      model_path="/home/felix/pointclouds/_working/2019_11_07_Muenchen_26_10_2018/sparse/1")

images = project.images
cameras = project.cameras

poses = []
dt = np.dtype([('x', np.float64), ('y', np.float64), ('z', np.float64), ('img', np.int32)])
for nr, img in images.items():
    R = img.qvec2rotmat()
    T = img.tvec
    cp = np.dot(-R.T, T)
    pose = (cp[0], cp[1], cp[2], nr)
    poses.append(pose)

vertex = np.array(poses, dtype=dt)
v = plyfile.PlyElement.describe(vertex, "vertices")
plyfile.PlyData([v], text=True).write(project.model_path + "/camera_locations.ply")
