from survey.sfm_helpers import Keypoint
from survey.detected_areas import DetectedArea
from survey.survey import Survey
from survey.helpers import Point2DType
import numpy as np


def RetrieveModelExample():
    #load survey
    survey_path = "/home/felix/pointclouds/katrin_isolate/my_first_survey/survey.ini"

    survey = Survey.LoadFromJson(survey_path)
    # which is the point im intrested in?
    looki_point = Keypoint("links/L1006.jpg", np.array([(0, 0)], dtype=Point2DType))

    # get area (e.g. construction site) this point belongs to
    area: DetectedArea
    area = survey.get_detected_area_2D(looki_point)
    # now work with this area its basically a proxy to the major_observation

    no_proxy = area.get_major_observation(survey)
    # local transforms cross_check will
    local_point = area.get_closest_feature(looki_point, survey, local=True, cross_check=False)
    pass
    # get the best view of this construction site (also happens to be where the local model ist located)


if __name__ == "__main__":
    RetrieveModelExample()
