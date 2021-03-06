from survey.geo_reference_information import CameraPoseGeoReferenceInformation
from survey.survey import Survey
from survey.commands import CreateLocalModels, CreateGlobalModel, ExtendDataBase, IsolateConstructionSites
from data_io import decoder_setup as serialization
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This tools prototype creates a complete resconstruction of an area as used (or it was applied) in the paper ...')
    parser.add_argument("-src", "--source_path", required=True, type=str, help="Folder of all images, with sub_folders")
    parser.add_argument("-svy", "--survey_path", required=True, type=str, help="Folder to create the full survey. Please check free memory it will probably be huge")
    parser.add_argument("-kpts", "--bounding_box_corners", required=True, type=str,
                        help="List of all bounding boxes that should be considered. One image AND keypoint per line, two keypoints (== two lines) are needed for every bounding box. See example files for more information")
    parser.add_argument("-gd", "--global_dense", action='store_true', help="Recursively walk all subdirectories", )

    # extract paths
    args = parser.parse_args()
    source_image_path = args.source_path
    survey_path = args.survey_path
    global_dense = args.global_dense
    bb_keypoints_file = args.bounding_box_corners

    survey = None
    # Check if survey exists else create new
    try:
        survey = Survey.LoadFromJson(survey_path + "/survey.ini")
    except:
        geo_info = CameraPoseGeoReferenceInformation()
        survey = Survey.CreateSurvey(source_image_path, survey_path, geo_info=geo_info, copy=True)
        survey.checkpoint()

    # Check checkpoint ... and ...
    if not survey.task_list.global_model:
        CreateGlobalModel(survey, with_dense=global_dense)
        survey.checkpoint()

    if not survey.task_list.construction_sites_isolated:
        ExtendDataBase(survey)
        IsolateConstructionSites(survey, bb_keypoints_file)
        survey.checkpoint()

    if not survey.task_list.local_models_created:
        CreateLocalModels(survey)
        survey.checkpoint()
