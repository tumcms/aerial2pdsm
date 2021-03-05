import shutil
from collections import namedtuple
from pathlib import Path
import numpy as np
import pymap3d as pm

from submodules.colmap_automate.app import ReconstructionConfig
from submodules.colmap_scripts.database import COLMAPDatabase
from sfm_helpers import Keypoint, SfmModel
from detected_areas import DetectedArea
from data_io.exif_adder import Gps2Exif
from geo_reference_information import GeoReferenceInformation, CameraPoseGeoReferenceInformation
from helpers import chmod_recursive, DistanceToRectangle, Point2DType
from observation import Observation
from data_io.serialization import Aerial2PdsmDecoder
from data_io.encoder_init import Aerial2PdsmEncoder
from data_io.sql_interface import GetObervationIdByImageName, GetAllAreas
from survey.tasklist import SurveyTaskList

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
            shutil.copytree(image_src, image_path, dirs_exist_ok=True)
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
        print(Aerial2PdsmEncoder._registered)
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



Aerial2PdsmEncoder.Register(ReconstructionConfig)
Aerial2PdsmEncoder.Register(CameraPoseGeoReferenceInformation)
Aerial2PdsmEncoder.Register(GeoReferenceInformation)
Aerial2PdsmEncoder.Register(SurveyTaskList)
Aerial2PdsmEncoder.Register(Survey)
