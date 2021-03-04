from pathlib import Path
import numpy as np
from survey import pdsm_creator
from data_io.closest_keypoint import ReadKeypointFile
from submodules.colmap_automate.app import CreateDirectory, Reconstructor, ReconstructionConfig
from submodules.colmap_scripts.database import COLMAPDatabase
from sfm_helpers import SfmModel, Keypoint
from survey.coordinate_mapping import CoordMapping
from survey.detected_areas import DetectedArea
from survey.geo_alignment import GeoAlignment
from survey.graph_analysis import CoveredAreaGraph
from survey.helpers import pairwise, GetClosestFeature, Unit, Project2Plane, RotateAroundAxis
from survey.observation import Observation, MedianFilter
from data_io.sql_interface import CreateObservationsTable, CreateDetectedAreasTable
from survey.survey import Survey


def CreateLocalModels(survey: Survey):
    # TODO add tasklist protection, it will rewrite all existing models
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


def CreateGlobalModel(survey: Survey, with_dense=False, force_new_model=False):
    if force_new_model:
        raise NotImplementedError("Sorry delete manually!")

    script_base_directory = Reconstructor.GetPathOfCurrentFile()
    source_config = script_base_directory.parent / "base_configurations" / "global_model"
    global_config_path = survey.get_global_config_path()
    Reconstructor.Generic2SpecificJobFiles(source_config, global_config_path, survey.rconfig_global)

    Reconstructor.execute_job(global_config_path / "1_extraction", survey.rconfig_global)

    # match without exhaustive
    if survey.rconfig_global.gps_available:
        Reconstructor.execute_job(global_config_path / "2_1_spatial_matcher", survey.rconfig_global)
        Reconstructor.execute_job(global_config_path / "2_2_transitive_matcher", survey.rconfig_global)
    else:
        Reconstructor.execute_job(global_config_path / "2_exhaustive_matcher", survey.rconfig_global)

    Reconstructor.execute_job(global_config_path / "3_mapper", survey.rconfig_global)

    # TODO handle multiple models, however there should only be one colmap model !
    Reconstructor.execute_job(global_config_path / "4_bundle_adjustment", survey.rconfig_global)
    Reconstructor.execute_job(global_config_path / "5_model_aligner", survey.rconfig_global)

    if with_dense:
        Reconstructor.execute_job(global_config_path / "6_image_undistorter", survey.rconfig_global)
        Reconstructor.execute_job(global_config_path / "7_patch_match", survey.rconfig_global)
        Reconstructor.execute_job(global_config_path / "8_stereo_fusion", survey.rconfig_global)

    survey.global_model = SfmModel(survey.get_global_model_path())
    survey.task_list.global_model = True

    # if not survey.task_list.global_model:
    #     raise AssertionError("Create a global model first!")


def ExtendDataBase(survey: Survey):
    db = COLMAPDatabase.connect(survey.get_database_path())
    CreateObservationsTable(db)
    CreateDetectedAreasTable(db)
    survey.task_list.extend_database = True


