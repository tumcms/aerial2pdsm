import shutil
from collections import namedtuple
from pathlib import Path

import numpy as np
import pymap3d as pm

import pdsm_creator
from closest_keypoint import ReadKeypointFile
from colmap_automate.app import ReconstructionConfig, CreateDirectory, Reconstructor
from colmap_scripts.database import COLMAPDatabase
from config import Keypoint, SfmModel
from detected_areas import DetectedArea
from exif_adder import Gps2Exif
from geo_alignment import GeoAlignment, GeoReferenceInformation, CameraPoseGeoReferenceInformation
from graph_analysis import CoveredAreaGraph
from helpers import pairwise, GetClosestFeature, Unit, Project2Plane, RotateAroundAxis, chmod_recursive, DistanceToRectangle, Point2DType
from observation import Observation, CoordMapping, MedianFilter
from serialization import Aerial2PdsmDecoder
from serialization import Aerial2PdsmEncoder
from sql_interface import CreateObservationsTable, CreateDetectedAreasTable, GetObervationIdByImageName, GetAllAreas

Aerial2PdsmEncoder.Register(ReconstructionConfig)
Aerial2PdsmEncoder.Register(CameraPoseGeoReferenceInformation)
Aerial2PdsmEncoder.Register(GeoReferenceInformation)


class SurveyTaskList:
    def __init__(self):
        self.global_model = False
        self.extend_database = False
        self.construction_sites_isolated = False
        self.construction_sites_mapped = False
        self.local_models_created = False
        self.mapping_2d_site_equipment = False
        self.point_cloud_analysis_tower_crane = False

    @classmethod
    def FromDict(cls, data):
        _class = cls()
        _class.__dict__ = data
        return _class

    def to_dict(self):
        d = self.__dict__
        d["type"] = self.__class__.__name__
        return d


Aerial2PdsmEncoder.Register(SurveyTaskList)

Candidate = namedtuple("Candidate", ["row", "distance"])


class Survey:

    def get_global_config_path(self):
        return self.root_path / "configurations" / "global_reconstruction"

    def get_image_path(self):
        return self.root_path / "images"

    def get_observations_path(self):
        return self.root_path / "observations"

    def get_detected_areas_path(self):
        return self.root_path / "detected_areas"

    def get_global_model_path(self):
        return self.root_path / "sparse" / "0"

    def __init__(self, survey_path, geo_info: GeoReferenceInformation, rconfig: ReconstructionConfig):
        self.root_path: Path = Path(survey_path)
        self.rconfig_global: ReconstructionConfig = rconfig
        self.task_list: SurveyTaskList = SurveyTaskList()
        self.geo_info: GeoReferenceInformation = geo_info
        self.global_model: SfmModel = None
        self.observations: {Observation}
        self.save_mode = True

    @staticmethod
    def CreateSurvey(image_src, survey_path, geo_info: GeoReferenceInformation, rconfig: ReconstructionConfig = None, copy=True):
        image_src = Path(image_src).expanduser()
        survey_path = Path(survey_path).expanduser()
        image_path = survey_path / "images"
        rconfig = ReconstructionConfig.CreateStandardConfig(survey_path, image_path, survey_path / survey_path.with_suffix(".db")) if not rconfig else rconfig

        if copy:
            shutil.copytree(image_src, image_path)
            chmod_recursive(image_path)

            # add gps information if available
            if isinstance(geo_info, CameraPoseGeoReferenceInformation):
                geo_info.ReadDlrCamPoses(image_path)

                if len(geo_info.poses) < 3:
                    raise FileNotFoundError("Geo reference are note sufficient. Provide more information")

                for pose in geo_info.poses.values():
                    lat, lon, alt = pm.ecef2geodetic(pose["x"], pose["y"], pose["z"])
                    Gps2Exif(image_path / str(pose["image"]), lon, lat, alt)
                rconfig.gps_available = True
                geo_info_path = survey_path / geo_info.get_std_name()
                geo_info.write(geo_info_path)
                rconfig.image_global_list = geo_info_path

        return Survey(survey_path, geo_info, rconfig)

    def __del__(self):
        if self.save_mode:
            self.checkpoint()

    def checkpoint(self):
        self.write(self.root_path / "survey.ini")

    def get_database_path(self):
        return self.rconfig_global.database_path

    def write(self, dest_path: Path = None):
        dest_path = Path(dest_path) if dest_path else self.root_path
        _json = Aerial2PdsmEncoder().encode(self)

    @staticmethod
    def LoadFromJson(path: Path):
        data = ""
        with open(path, "r") as f:
            data = f.read()
        decoded = Aerial2PdsmDecoder().decode(data)
        decoded.root_path = Path(decoded.root_path)

        # geo_info = GeoReferenceInformation.FromDict(data["geo_info"])
        # rconfig = ReconstructionConfig.FromDict(data["rconfig_global"])
        # survey_path = Path(data["root_path"])
        #
        # survey = Survey(survey_path, geo_info, rconfig)
        # survey.task_list = SurveyTaskList.FromDict(data["task_list"])
        # survey.global_model = SfmModel(data["model_path"])
        # load observations ? self.observations: {Observation}
        return decoded

    def to_dict(self):
        d = self.__dict__
        d["type"] = self.__class__.__name__
        return d

    def get_detected_area_2D(self, keypoint: Keypoint) ->DetectedArea:
        observation_rows = None
        image_name = keypoint.image
        point = keypoint.point

        with COLMAPDatabase.connect(self.get_database_path()) as db:
            observation_rows = GetObervationIdByImageName(db, image_name)

        closest_match = Candidate([], 1e10)
        for row in observation_rows:
            corners = np.array([(row[2], row[3]), (row[4], row[5])], dtype=Point2DType)
            dist = DistanceToRectangle(corners, point)

            if dist < closest_match.distance:
                closest_match = Candidate(row, dist)

        area = DetectedArea.GetAreaById(self, closest_match.row[1])
        if area:
            return area
        return None
        # return closest_match.row[0]

    def get_all_areas(self):
        with COLMAPDatabase.connect(self.get_database_path()) as db:
            area_rows = GetAllAreas(db)
        areas = []
        for row in area_rows:
            areas.append(DetectedArea._ConstructFromDB(row))
        return areas


Aerial2PdsmEncoder.Register(Survey)


def CreateGlobalModel(survey: Survey, with_dense=False, force_new_model=False):
    if force_new_model:
        raise NotImplementedError("Sorry delete manually!")

    script_base_directory = Reconstructor.GetPathOfCurrentFile()
    source_config = script_base_directory.parent / "base_configurations" / "global_model"
    global_config_path = survey.get_global_config_path()

    Reconstructor.Generic2SpecificJobFiles(source_config, global_config_path, survey.rconfig_global)

    # TODO uncomment
    Reconstructor.execute_job(global_config_path / "1_extraction", survey.rconfig_global)
    # match without exhaustive
    if survey.rconfig_global.gps_available:
        Reconstructor.execute_job(global_config_path / "2_1_spatial_matcher", survey.rconfig_global)
        Reconstructor.execute_job(global_config_path / "2_2_transitive_matcher", survey.rconfig_global)
    else:
        Reconstructor.execute_job(global_config_path / "2_exhaustive_matcher", survey.rconfig_global)

    Reconstructor.execute_job(global_config_path / "3_mapper", survey.rconfig_global)

    # TODO handle multiple models
    Reconstructor.execute_job(global_config_path / "4_bundle_adjustment", survey.rconfig_global)
    Reconstructor.execute_job(global_config_path / "5_model_aligner", survey.rconfig_global)

    if with_dense:
        Reconstructor.execute_job(global_config_path / "6_image_undistorter", survey.rconfig_global)
        Reconstructor.execute_job(global_config_path / "7_patch_match", survey.rconfig_global)
        Reconstructor.execute_job(global_config_path / "8_stereo_fusion", survey.rconfig_global)

    survey.global_model = SfmModel(survey.get_global_model_path())
    survey.task_list.global_model = True

    if not survey.task_list.global_model:
        raise AssertionError("Create a global model first!")

    # try:
    #     db = COLMAPDatabase.connect(survey.get_database_path())
    #     db.execute(create)
    #     image_row = db.execute("SELECT image_id FROM images WHERE name=?", c00.image).fetchone()
    #     db.close()


def ExtendDataBase(survey: Survey):
    db = COLMAPDatabase.connect(survey.get_database_path())
    CreateObservationsTable(db)
    CreateDetectedAreasTable(db)
    survey.task_list.extend_database = True


def IsolateConstructionSites(survey: Survey, object_detection_keypoint_file_path):
    """
    # This will first calculate and anaylse all images called observations, after this, the things will be matched and
    # grouped as detected areas

    :param survey:
    :param object_detection_keypoint_file_path:
    """
    if not survey.global_model.cameras:
        SfmModel.parse_binary_model(survey.global_model)

    if not survey.task_list.extend_database:
        raise ConnectionError("Database was not extended previously")

    keypoint_file_path = Path(object_detection_keypoint_file_path)
    keypoints = ReadKeypointFile(keypoint_file_path)
    CreateDirectory(survey.get_observations_path())
    observation_graphs = []

    for i00, i11 in pairwise(keypoints):
        m00 = GetClosestFeature(i00, survey.global_model)
        if not m00.valid: continue
        m11 = GetClosestFeature(i11, survey.global_model)
        if not m11.valid: continue
        image_id = m00.image.id
        observation = Observation(image_id, m00.image.name, i00, i11, m00, m11)
        observation.image_name = i00.image
        # Get Z Axis with the normal of the WGS-84
        wgs_m00 = pm.ecef2geodetic(m00.point3d.xyz[0], m00.point3d.xyz[1], m00.point3d.xyz[2])
        zdiff = wgs_m00[2] - 50
        mxx = pm.geodetic2ecef(wgs_m00[0], wgs_m00[1], zdiff)
        z_axis = Unit(np.array([m00.point3d.xyz[0] - mxx[0], m00.point3d.xyz[1] - mxx[1], m00.point3d.xyz[2] - mxx[2]]))

        # project the vector from point m00 -> m11 to the plane created by the z-vector
        hyp_3d = Project2Plane(m11.point3d.xyz - m00.point3d.xyz, z_axis)
        observation.hyp_3d = hyp_3d

        # Get X Axis by using the angle between 2d X & hypotenuse in z-plane
        i10 = Keypoint(i11.image, np.array([i11.point[0], i00.point[1]]))  # image space, right point
        hyp_2d = i11.point - i00.point  # image space, diagonal
        ref_2d = i10.point - i00.point  # image space, x-axis
        theta = -1 * np.arccos(np.dot(ref_2d, hyp_2d) / (np.linalg.norm(ref_2d) * np.linalg.norm(hyp_2d)))  # angle hyp and ref

        # Create Pointed Lines for visualisation
        x_axis = Unit(RotateAroundAxis(hyp_3d, z_axis, theta))
        y_axis = Unit(np.cross(z_axis, x_axis))

        image_base = np.zeros((4, 4))
        image_base[:3, :3] = np.column_stack([x_axis, y_axis, z_axis]).T
        m00_in_rotated = np.dot(image_base[:3, :3], m00.point3d.xyz)
        image_base[:3, 3] = np.array([-m00_in_rotated])
        image_base[3, 3] = 1
        # create mapping global ecef => local with m00 as base
        observation.mapping = CoordMapping.CreateLocalMapping(image_base)
        # Get all points that are inside the rectangle
        observation.inliers = observation.filter_inside(observation.mapping, survey.global_model)

        # update base with a appropriate z-height to best fit the ground
        observation.mapping.image_base[2, 3] = -np.dot(image_base[:3, :3], MedianFilter(observation.inliers).xyz)[2]
        observation.geo_hash = GeoAlignment.CreateGeoHash(observation.get_global_center())
        images_area, max_matches = observation.matches_per_image(survey.global_model)
        observation.graph = CoveredAreaGraph("ConstructionSite", observation.get_short_image_name(), images_area)
        observation_graphs.append(observation.graph)

        # Write Observation to disk & obtaining a true ID
        with COLMAPDatabase.connect(survey.get_database_path()) as db:
            observation.id = observation.write2database(db)
            observation.graph.graph["observation_id"] = observation.id
            observation.write2aux(survey.get_image_path(), survey.get_observations_path() / observation.get_relative_path())

        # points_in_rect, max_features = CountMatches(project, points_within_site)
        # print("Points on CS: {0}".format(len(points_in_rect)))
        #

    detected_areas = DetectedArea.GroupObservations(observation_graphs)

    area: DetectedArea
    for area in detected_areas:
        with COLMAPDatabase.connect(survey.get_database_path()) as db:
            area.write2database(db)
        area.write2xml(survey.get_detected_areas_path())

    survey.task_list.construction_sites_isolated = True
    survey.task_list.construction_sites_mapped = True


def CreateLocalModels(survey: Survey):
    # TODO add tasklist protextion
    if not survey.global_model.cameras:
        SfmModel.parse_binary_model(survey.global_model)

    areas: DetectedArea
    for areas in survey.get_all_areas():
        major_observation = areas.get_major_observation(survey)
        new_base_path = survey.get_observations_path() / major_observation.get_relative_path() / major_observation.get_local_model_path()
        points_inside = major_observation.filter_inside(major_observation.mapping, survey.global_model)

        # Exctract a Model with images only, the ids will be continues
        sub_model = pdsm_creator.ExtractSubModel(survey.global_model, points_inside, new_base_path)
        min_depth, max_depth = pdsm_creator.AnalyseDepth(sub_model)

        CreateDirectory(Path(sub_model.base_path))
        # Copies needed images
        sub_model.images_path = sub_model.base_path / "images"
        pdsm_creator.PortImages(survey.global_model, survey.get_image_path(), sub_model)
        pdsm_creator.AnalyseDepth(sub_model)
        # sets up the reconstruction
        this_file_path = Reconstructor.GetPathOfCurrentFile()
        source_config = this_file_path.parent / "base_configurations" / "local_model"
        rec_conf = ReconstructionConfig.CreateStandardConfig(sub_model.base_path, database_path=Path(sub_model.base_path, "pdsm.db"),
                                                             min_depth=min_depth - 10, max_depth=max_depth)
        rec_conf.image_global_list = survey.rconfig_global.image_global_list
        dest_config = Path(sub_model.base_path, "local_configuration")

        # This might look like a little freeze, but its actually taking time for the models.
        Reconstructor.Generic2SpecificJobFiles(source_config, dest_config, rec_conf)

        Reconstructor.execute_all(dest_config, rec_conf)

        # This would sync the model to a reconstruction. This will be skipped due to colmap removing some important points
        # sub_model = pdsm_creator.SyncModelWithDatabase(sub_model)
        # pdsm_creator.SaveModel(sub_model)
        # pdsm_creator.ReconstructModel(sub_model, folder_path + "/dense_site.ply")

    survey.task_list.local_models_created = True
