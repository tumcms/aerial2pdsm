import xml.etree.ElementTree as ET

import numpy as np

from colmap_automate.app import CreateDirectory
from colmap_scripts.database import COLMAPDatabase
from config import Keypoint, SfmModel
from helpers import GetClosestFeature
from observation import Observation, CoordMapping
from sql_interface import InsertIntoDetectedAreas, GetDetectedAreaById, GetObservationById, UpdateObservation_Area


class DetectedArea:

    def __init__(self, obs_ids, graphs=[]):
        self.id = -1
        self.observations_ids = obs_ids
        self.graphs = graphs
        self.major_observation = None

    def get_major_observation_id(self):
        return next(iter(self.observations_ids), -1)

    def write2database(self, db):
        self.id = InsertIntoDetectedAreas(db, self.get_major_observation_id(), self.observations_ids)
        for obs_id in self.observations_ids:
            UpdateObservation_Area(db, self.id, obs_id)
        return self.id

    def get_xml_path(self):
        return "{}_equivalent_constructionsites.xml".format(self.id)

    def get_major_observation(self, survey) -> Observation:
        if not self.major_observation:
            with COLMAPDatabase.connect(survey.get_database_path()) as db:
                data = GetObservationById(db, self.get_major_observation_id())

            i00 = Keypoint(data[17], np.array([data[3], data[4]]))
            i11 = Keypoint(data[17], np.array([data[5], data[6]]))
            m00 = GetClosestFeature(i00, survey.global_model)
            m11 = GetClosestFeature(i11, survey.global_model)
            obs = Observation(data[16], data[17], i00, i11, m00, m11)
            obs.id = data[0]

            # could leverage from db
            # rotm = qvec2rotmat(np.array([data[8], data[9], data[10], data[11]]))
            # tvec = np.array(data[12], data[13], data[14])
            # or out file!
            imagebase_path = survey.get_observations_path() / obs.get_relative_path() / obs.get_imagebase_filename()
            obs.mapping = CoordMapping(np.loadtxt(imagebase_path))
            self.major_observation = obs
        return self.major_observation

    def write2xml(self, outpath):
        CreateDirectory(outpath)
        root = ET.Element("observations")
        # group = ET.SubElement(root, "Group_{}".format("1"))
        for cs in self.graphs:
            csite = ET.SubElement(root, str(cs.name))
            features = ET.SubElement(csite, "features").text = str(cs.feature_count)
            nrcont = ET.SubElement(csite, "number_of_contributing_images").text = str(cs.value - 1)
            cont_img = ET.SubElement(csite, "images")

            iter_img = iter(cs.nodes)
            next(iter_img)  # remove root

            for cimg in iter_img:
                ET.SubElement(cont_img, cimg)

        tree = ET.ElementTree(root)
        tree.write(outpath / self.get_xml_path())

    @staticmethod
    def GroupObservations(observations_graph_iter):
        # Group Observations
        detected_areas = []
        observations_graph_iter.sort(key=len, reverse=True)
        # graphnr = 0
        for graph_cnt, observation_graph in enumerate(observations_graph_iter):
            found = False
            observation_graph.number = graph_cnt  # graphnr
            # graphnr += 1
            observation_graph.value = len(observation_graph)
            observation_graph.feature_count = sum([n[2]["weight"] for n in observation_graph.edges(data=True)])
            observation_graph.mean = -1

            for exist in detected_areas:
                group_similarity = []
                for edge in observation_graph.edges("ConstructionSite"):
                    base_node = edge[0]
                    far_node = edge[1]
                    e2 = exist[0].get_edge_data(base_node, far_node)
                    if e2:
                        e1 = observation_graph.get_edge_data(base_node, far_node)
                        w1 = e1["weight"] / observation_graph.feature_count
                        w2 = e2["weight"] / exist[0].feature_count
                        diff = w2 - w1
                        ndiff = diff / w1 if diff > 0 else diff / w2
                        group_similarity.append(abs(ndiff))
                    else:
                        group_similarity.append(1)

                mean = sum(group_similarity) / len(group_similarity)
                if mean < 0.4:
                    observation_graph.value = len(observation_graph)
                    exist.append(observation_graph)
                    exist.sort(key=len, reverse=True)
                    found = True
                    observation_graph.mean = mean
                    break
            if not found:
                observation_graph.value = len(observation_graph)
                observation_graph.mean = 0
                detected_areas.append([observation_graph])

        areas: [DetectedArea] = []
        for graph_list in detected_areas:
            areas.append(DetectedArea([g.graph["observation_id"] for g in graph_list], graph_list))
        return areas

    @staticmethod
    def GetAreaById(survey, deta_id):
        with COLMAPDatabase.connect(survey.get_database_path()) as db:
            data = GetDetectedAreaById(db, deta_id)
        if data:
            return DetectedArea._ConstructFromDB(data)

    @staticmethod
    def _ConstructFromDB(data):
        obs_ids = [int(s) for s in data[2].split(",")]
        area = DetectedArea(obs_ids)
        area.id = int(data[0])
        return area

    def get_closest_feature(self, keypoint, survey, local=True, cross_check=False):
        major = self.get_major_observation(survey)
        local_model_path = survey.get_observations_path() / major.get_relative_path() / major.get_local_model_path()
        if not hasattr(major, "local_model"):
            major.local_model = SfmModel(local_model_path / "sparse")
            major.local_model.parse_binary_model()

        feature = GetClosestFeature(keypoint, major.local_model)
        if cross_check:
            rough_feature = GetClosestFeature(keypoint, survey.global_model)
            if rough_feature.distance < feature.distance:
                feature = rough_feature

        if local:
            local_xyz = major.mapping.global2local(feature.point3d.xyz)
            feature.point3d = feature.point3d._replace(xyz=local_xyz)
        return feature
