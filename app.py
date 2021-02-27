from geo_alignment import CameraPoseGeoReferenceInformation
from helpers import Point2DType
from survey import Survey, CreateLocalModels, CreateGlobalModel, ExtendDataBase, IsolateConstructionSites
import numpy as np

if __name__ == "__main__":
    # TODO Parse Arguments

    # TODO If this is correct, how do i get the paths right? Relative is correct but thats kinda mähhhh
    source_image_path = "/home/felix/pointclouds/katrin_isolate/2018-10-02/Reconstruction_Muenchen_10_02_2018/images"
    survey_path = "/home/felix/pointclouds/katrin_isolate/my_first_survey"

    survey = None
    try:
        survey = Survey.LoadFromJson(survey_path + "/survey.ini")
    except:
        geo_info = CameraPoseGeoReferenceInformation()
        survey = Survey.CreateSurvey(source_image_path, survey_path, geo_info=geo_info, copy=True)
        survey.checkpoint()

    # if not survey.task_list.global_model:
    #     CreateGlobalModel(survey)
    #     survey.checkpoint()

    if not survey.task_list.construction_sites_isolated:
        ExtendDataBase(survey)
        IsolateConstructionSites(survey, "/home/felix/pointclouds/katrin_isolate/2018-10-02/2018-10-02.txt")
        survey.checkpoint()

    if not survey.task_list.local_models_created:
        CreateLocalModels(survey)
        survey.checkpoint()


