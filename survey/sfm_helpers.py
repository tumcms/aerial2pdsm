from pathlib import Path
from colmap_scripts.read_write_model import Point3D, Image, Camera
from data_io.encoder_init import Aerial2PdsmEncoder
from colmap_scripts import read_write_model as rm
import numpy as np


class QueryMatch:
    def __init__(self):
        self.image = None
        self.point2d_id = -1
        self.point2d_coord = [0, 0]
        self.point3d_id = -1
        self.point3d = rm.Point3D
        self.query_point2d = [0, 0]
        self.distance = -1
        self.valid = True


class Keypoint:
    def __init__(self, img_name="", vec=np.array([0, 0])):
        self.image = img_name
        self.point = vec


class SfmModel:

    def __init__(self, path: Path):
        self.path = path
        self.points: {Point3D} = {}
        self.images: {Image} = {}
        self.cameras: {Camera} = {}

    def parse_binary_model(self):
        if not (self.cameras and self.images and self.points):
            self.cameras = rm.read_cameras_binary(self.path / "cameras.bin")
            self.images = rm.read_images_binary(self.path / "images.bin")
            self.points = rm.read_points3d_binary(self.path / "points3D.bin")

    def to_dict(self):
        d = self.__dict__
        d["type"] = self.__class__.__name__
        try:
            del d["points"]
            del d["images"]
            del d["cameras"]
        except:
            # No points loaded
            pass
        return d

    @staticmethod
    def FromDict(dct):
        obj = SfmModel(Path(dct["path"]))
        return obj


class SparseModel(SfmModel):

    def __init__(self, base_path, model_path="", images_path="", db_path=""):
        base_path = Path(base_path)
        super().__init__(base_path)
        self.base_path = base_path
        self.model_path = model_path if model_path else base_path / r"/sparse/aligned"
        self.images_path = images_path if images_path else base_path / r"/images"
        self.database_path = db_path if db_path else base_path / r"/database.db"
        self.dat_path = base_path / r"/aux"

    def parse_binary_model(self, model=None):
        SfmModel.parse_binary_model(self)


def GetInlineMatches(img: rm.Image):
    return img.xys[img.point3D_ids != -1], img.point3D_ids[img.point3D_ids != -1]


Aerial2PdsmEncoder.Register(SfmModel)
Aerial2PdsmEncoder.Register(SparseModel)
