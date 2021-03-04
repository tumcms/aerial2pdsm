from pathlib import Path
import numpy as np
from sfm_helpers import SfmModel


class CoordMapping:

    def __init__(self, image_base):
        self.image_base = image_base

    @staticmethod
    def GetArea3D(p1, p2):
        area = p1[1] * p2[0] - p1[0] * p1[0] - p2[1] * p1[0] - p2[1] * p1[0]
        return area

    def get_axoriented_bounding_cell(self, p0, p1, stay_global=False):
        if not stay_global:
            nb00 = self.global2local(p0)
            nb11 = self.global2local(p1)
        else:
            nb00 = np.dot(self.rotm(), p0)
            nb11 = np.dot(self.rotm(), p1)

        minx = min(nb00[0], nb11[0])
        miny = min(nb00[1], nb11[1])
        maxx = max(nb00[0], nb11[0])
        maxy = max(nb00[1], nb11[1])
        # return np.array([(minx,), (miny,), (maxx,), (maxy,)], dtype=Boundaries2D)
        return np.array([minx, miny, maxx, maxy])

    def rotm(self):
        return self.image_base[:3, :3]

    def trav(self):
        return self.image_base[:3, 3]

    def local2global(self, local_point):
        return np.dot(np.invert(self.image_base), np.append(local_point, 1))
        # return np.dot(self.image_base[:3, :3].T, local_point) - self.image_base[:3, 3]

    def global2local(self, global_point):
        # return np.dot(self.image_base, np.append(global_point, 1))
        return np.dot(self.image_base[:3, :3], global_point) + self.image_base[:3, 3]

    def translate(self, transv):
        self.image_base[:3, 3] -= transv

    # From docu => Point3D = collections.namedtuple( "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
    @staticmethod
    def InsideArea(global_transform, sfm: SfmModel, coordinates="ecef"):
        filtered = []
        bounds = global_transform.GetMinMax()
        rotm = global_transform.image_base[:3, :3].T
        trav = global_transform.image_base[:3, 3]
        minx = bounds.min_x + trav[0]
        maxx = bounds.max_x + trav[0]
        miny = bounds.min_y + trav[1]
        maxy = bounds.max_y + trav[1]

        for p3d in sfm.points.values():
            p = np.dot(rotm, p3d.xyz)
            if minx < p[0] < maxx:
                if miny < p[1] < maxy:
                    filtered.append(p3d)
        return filtered

    @staticmethod
    def CreateLocalMapping(image_base):
        mapping = CoordMapping(image_base)
        return mapping

    def Write(self, path):
        path = Path(path)
        np.savetxt(path, self.image_base, fmt='%f')

    # np.column_stack([x_axis, y_axis, z_axis]).T
    def get_xaxis(self):
        return self.image_base[:3, 1]

    def get_yaxis(self):
        return self.image_base[:3, 2]

    def get_zaxis(self):
        return self.image_base[:3, 3]