from pathlib import Path
from matplotlib import patches

from colmap_automate.app import CreateDirectory
from colmap_scripts.read_write_model import rotmat2qvec, Point3D
from sfm_helpers import QueryMatch, Keypoint, SfmModel
import numpy as np

from coordinate_mapping import CoordMapping
from graph_analysis import WriteEgoGraph
from helpers import CreateLine
from plots.plotter import PlotNext
from data_io.serialization import WritePly
from data_io.sql_interface import InsertIntoObservations
import matplotlib.pyplot as plt

Boundaries2D = np.dtype([
    ('minx', np.float64),
    ('miny', np.float64),
    ('maxx', np.float64),
    ('maxy', np.float64)
])


def MedianFilter(points):
    p = list(points)
    tmp = sorted(p, key=lambda p: p.xyz[2])
    median_i = int(len(tmp) / 2)
    return tmp[median_i]


class Observation:

    def __init__(self, image_id, image_name, i00: Keypoint, i11: Keypoint, m00: QueryMatch, m11: QueryMatch, hypo=None):
        self.id = -1
        self.image_id = image_id
        self.image_name = image_name
        self.m00 = m00
        self.m11 = m11
        self.i00 = i00
        self.i11 = i11
        self.graph = ""
        self.geo_hash = 0
        self.mapping: CoordMapping = None
        self.inliers = None
        self.images_and_matches = {}
        self.hyp_3d = hypo

    def get_short_image_name(self):
        return Path(self.image_name).with_suffix("").name

    def get_global_center(self):
        return (self.m00.point3d.xyz + self.m11.point3d.xyz) / 2

    def get_bounding_box2D(self):
        return np.column_stack(self.m00.point2d_coord, self.m11.point2d_coord)

    def write2database(self, db):
        qvec = rotmat2qvec(self.mapping.rotm())
        tvec = np.dot(-self.mapping.rotm(), self.mapping.trav())

        row_id = InsertIntoObservations(db, self.image_id, self.i00, self.i11, self.geo_hash, qvec, tvec,
                                        self.mapping.GetArea3D(self.m00.point3d.xyz, self.m00.point3d.xyz))
        return row_id

    def filter_inside(self, mapping: CoordMapping, sfm: SfmModel):
        dat = []
        bounds = mapping.get_axoriented_bounding_cell(self.m00.point3d.xyz, self.m11.point3d.xyz, stay_global=True)
        for p3d in sfm.points.values():
            p = np.dot(mapping.rotm(), p3d.xyz)
            if bounds[0] <= p[0] <= bounds[2]:
                if bounds[1] <= p[1] <= bounds[3]:
                    dat.append(p3d)
        return dat

    @staticmethod
    def CoordinateTransform(mapping, point_list: [Point3D]):
        nr_points = len(point_list)
        transformed = [None] * nr_points
        if nr_points > 0 & isinstance(point_list[0], Point3D):
            for i, p3d in enumerate(point_list):
                transformed[i] = p3d._replace(xyz=mapping.global2local(p3d.xyz))
            return transformed
        else:
            for i, p3d in enumerate(point_list):
                transformed[i] = mapping.global2local(p3d)
            return transformed

    def get_local_model_path(self):
        return Path("local_model")

    def matches_per_image(self, project: SfmModel, threshold=0.1):
        uimages = {}
        # Count number of inline features of the data source (mostly a selected area)
        for p3d in self.inliers:
            for iid in p3d.image_ids:
                full_path = project.images[iid.item()].name
                name = full_path.split("/")[1]
                name = name.split(".")[0]
                if name not in uimages:
                    uimages[name] = {"val": 0, "full_path": full_path, "id": iid}
                uimages[name]["val"] += 1
        highest_number_of_matches = 0

        # Determine the highest number of features that are seen by any image
        for iname, img in uimages.items():
            cval = img["val"]
            if cval > highest_number_of_matches:
                highest_number_of_matches = cval

        # remove all that are lower than the filter criteria
        remove = [k for k, c in uimages.items() if c["val"] < threshold * highest_number_of_matches]
        for k in remove: del uimages[k]

        return uimages, highest_number_of_matches

    def get_relative_path(self):
        # TODO somehow manage all the paths
        return Path("{}_{}".format(self.id, self.get_short_image_name()))

    def get_imagebase_filename(self):
        return "#{}_local2global.out".format(self.id)

    def write2aux(self, image_path, result_folder):
        obs_id = self.id
        image_name = self.get_short_image_name()

        if obs_id == -1:
            raise AssertionError("Id not set")

        CreateDirectory(result_folder)

        # Point clouds and transformations
        global_sparse = result_folder / "#{}_sparse_global.ply".format(obs_id)
        local_sparse = result_folder / "#{}_sparse_cssystem.ply".format(obs_id)
        local2global = result_folder / self.get_imagebase_filename()
        WritePly(self.inliers, global_sparse)
        WritePly(Observation.CoordinateTransform(self.mapping, self.inliers), local_sparse, precision="float")
        CoordMapping.Write(self.mapping, local2global)

        # Axis for debug and interest
        x_axis_global = result_folder / "#{}_xaxis.ply".format(obs_id)
        y_axis_global = result_folder / "#{}_yaxis.ply".format(obs_id)
        z_axis_global = result_folder / "#{}_zaxis.ply".format(obs_id)
        diagonal_global = result_folder / "#{}_diagonal3D.ply".format(obs_id)

        WritePly(CreateLine(self.m00.point3d.xyz, self.mapping.get_xaxis() * 50), x_axis_global)
        WritePly(CreateLine(self.m00.point3d.xyz, self.mapping.get_yaxis() * 50), y_axis_global)
        WritePly(CreateLine(self.m00.point3d.xyz, self.mapping.get_zaxis() * 50), z_axis_global)
        if self.hyp_3d is not None:
            WritePly(CreateLine(self.m00.point3d.xyz, self.m00.point3d.xyz + self.hyp_3d), diagonal_global)

        construction_site_graph = result_folder / "#{}_graph.ply".format(obs_id)
        if self.graph:
            WriteEgoGraph(self.graph, "ConstructionSite", construction_site_graph)

        image_detected_area = result_folder / "#{}_features.svg".format(self.id)

        plot = plt.figure()
        plot.sub = []
        # TODO fix that mplotlib ting
        PlotNext(image_path, plot, [self.m00, self.m11], nopoints=False, size=0.5, extra_size=2)
        diff = self.m11.point2d_coord - self.m00.point2d_coord
        rect = patches.Rectangle((self.m00.point2d_coord[0], self.m00.point2d_coord[1]), diff[0], diff[1], linewidth=1, edgecolor='c', facecolor='none')
        plot.sub[-1].add_patch(rect)
        plt.savefig(image_detected_area, dpi=400)


