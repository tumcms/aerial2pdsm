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