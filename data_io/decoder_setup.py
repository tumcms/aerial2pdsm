from submodules.colmap_automate.app import ReconstructionConfig
from survey.geo_reference_information import GeoReferenceInformation, CameraPoseGeoReferenceInformation
from encoder_init import Aerial2PdsmEncoder
from survey.survey import Survey
from survey.tasklist import SurveyTaskList

def init():
    Aerial2PdsmEncoder.Register(ReconstructionConfig)
    Aerial2PdsmEncoder.Register(CameraPoseGeoReferenceInformation)
    Aerial2PdsmEncoder.Register(GeoReferenceInformation)
    Aerial2PdsmEncoder.Register(SurveyTaskList)
    Aerial2PdsmEncoder.Register(Survey)
    print("classes registered")