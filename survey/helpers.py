import numpy as np
import plyfile # can also be used from colmap_scripts
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from submodules.colmap_scripts import read_write_model as rm
from submodules.colmap_scripts.read_write_model import Point3D
from sfm_helpers import QueryMatch, Keypoint, SparseModel, GetInlineMatches, SfmModel
from scipy.spatial import KDTree, cKDTree
from os.path import basename
import os


# https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
def closest_point2d(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def GetClosestFeature(query_point: Keypoint, model: SfmModel) -> QueryMatch:
    # img: rm.Image = next((i for i in model.images.values() if i.name == query_point.image), None)
    qname = basename(query_point.image)
    img = next((i for i in model.images.values() if qname in i.name), None)
    cfeature = QueryMatch()
    if not img:
        cfeature.valid = False
    else:
        cur_points2d_coord, cur_points3d_ids = GetInlineMatches(img)
        cfeature = QueryMatch()
        cfeature.image = img
        cfeature.query_point2d = query_point.point
        cfeature.point2d_id = closest_point2d(query_point.point, cur_points2d_coord)
        cfeature.point3d_id = cur_points3d_ids[cfeature.point2d_id]
        cfeature.point3d = model.points[cfeature.point3d_id]
        cfeature.point2d_coord = cur_points2d_coord[cfeature.point2d_id]
        cfeature.distance = np.linalg.norm(cfeature.point2d_coord - query_point.point)
    return cfeature


def FastLocationSearch(img_name, qps: np.array, model: SparseModel) -> [QueryMatch]:
    # img: rm.Image = next((i for i in model.images.values() if i.name == img_name), None)
    img = next((i for i in model.images.values() if i.name == img_name), None)
    feature2d_coords, feature2d_ids = GetInlineMatches(img)
    fvec = np.reshape(np.asarray(feature2d_coords, dtype=np.int64), (-1, 2))
    tree = cKDTree(fvec, leafsize=100)
    nn_d, nn_i = tree.query(qps)
    return [nn_i, nn_d]


def pairwise(it):
    it = iter(it)
    try:
        while True:
            yield next(it), next(it)
    except StopIteration:
        return


def Unit(v): return v / np.linalg.norm(v)


def Project2Plane(vec, plane):
    return vec - np.dot(vec, plane) * plane


def MultiplyQuaternion(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def RotateAroundAxis(origin, axis, angle):
    q1 = np.pad(origin, (1, 0))
    _t = angle / 2
    cos_t = np.cos(_t)
    sin_t = np.sin(_t)
    r = Unit(axis)
    q2 = np.array([cos_t, r[0] * sin_t, r[1] * sin_t, r[2] * sin_t])
    q2_conj = np.array([cos_t, -r[0] * sin_t, -r[1] * sin_t, -r[2] * sin_t])

    q3 = MultiplyQuaternion(q2, MultiplyQuaternion(q1, q2_conj))
    return q3[1:]


def CalculateArea(a, b):
    return np.abs(np.cross(a, b))


def PointsToPly(pcloud: [Point3D], filename="some_file.ply"):
    try:
        vertex = np.array([i.xyz for i in pcloud])  # np.stack([mx, my, mz, m00.point3d.xyz])
    except:
        vertex = np.array([i for i in pcloud])

    vertex.dtype = [("x", "f8"), ("y", "f8"), ("z", "f8")]
    vertex = vertex.squeeze(axis=1)
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
    plyfile.PlyData([v], text=True).write(filename)


def CreateLine(p1, p2):
    diag_interp_x = np.linspace(p1[0], p2[0], 100)
    diag_interp_y = np.linspace(p1[1], p2[1], 100)
    diag_interp_z = np.linspace(p1[2], p2[2], 100)
    diag3d_points = list(zip(diag_interp_x, diag_interp_y, diag_interp_z))
    return diag3d_points  # PointsToPly(diag3d_points, path_to_plyfile)


def InvertTransformation(trafo_matrix):
    inv = np.zeros((4, 4))
    inv[:3, :3] = trafo_matrix[:3, :3].T
    inv[:3, 3] = - np.dot(inv[:3, :3], trafo_matrix[:3, 3])
    inv[3, 3] = 1
    return inv


def chmod_recursive(path, permission=0o755):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), permission)
        for f in files:
            os.chmod(os.path.join(root, f), permission)


def general_to_dict(self):
    d = dict()
    for a, v in self.__dict__.items():
        if hasattr(v, "to_dict"):
            d[a] = v.to_dict()
        else:
            d[a] = v
    return d


Point2DType = np.type = [("x", "i4"), ("y", "i4")]


def DistanceToRectangle(corners: np.array, point):
    c = structured_to_unstructured(corners)
    point = structured_to_unstructured(point)

    delta = np.max(np.hstack([np.min(c, axis=0) - point, np.array([0, 0]).reshape(1, 2), point - np.max(c, axis=0)]))
    return np.linalg.norm(delta)

    # dx = np.max(np.min(corners, order="x") - point[0], 0, point[0] - np.max(corners, order="x"))
    # dy = np.max(np.min(corners, order="y") - point[1], 0, point[1] - np.max(corners, order="y"))
    # return np.sqrt(dx*dx + dy * dy)
