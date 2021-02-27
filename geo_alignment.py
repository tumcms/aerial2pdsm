from pathlib import Path
import config
from re import compile
import numpy as np
import pymap3d as pm

from serialization import Aerial2PdsmEncoder

CameraPose = np.dtype([('image', "U32"), ('x', np.float64), ('y', np.float64), ('z', np.float64)])

from colmap_scripts.database import COLMAPDatabase

pattern_read = compile(r'(^\w*).(\w*):\s*([\w\d,\[\]\.]*)')
pattern_write = compile(r'.*\[([\d\.]*),([\d\.]*),([\d\.]*)\]')


def ReadAux_DLR(aux_path):
    image_dat = {}
    with open(aux_path, "r") as meta:
        for line in meta.readlines():
            try:
                item, prop, value = pattern_read.findall(line)[0]
                if item not in image_dat:
                    image_dat[item] = {}
                image_dat[item][prop] = value
            except ValueError:
                pass

    return image_dat


class GeoReferenceInformation:

    def ReadDlrCamPoses(self, source_path, ECEF=False):
        source_path = Path(source_path)

        img_aux = GeoAlignment.ExtractAuxData(source_path, ".[Aa][Uu][Xx]")
        for img_path, aux_path in img_aux.items():
            dat = ReadAux_DLR(aux_path)
            if not ECEF:
                lon, lat, alt = pattern_write.findall(dat["GPSIMU"]["Position"])[0]
                x, y, z = pm.geodetic2ecef(float(lat), float(lon), float(alt))
            else:
                x, y, z = pattern_write.findall(dat["Sensor"]["Position"])[0]
            rel_img_path = img_path.relative_to(source_path)
            self.poses[str(rel_img_path)] = np.array((str(rel_img_path), x, y, z), dtype=CameraPose)
        return self

    def to_dict(self):
        d = self.__dict__
        d["type"] = self.__class__.__name__
        return d


# e.g. GPS during recording
class CameraPoseGeoReferenceInformation(GeoReferenceInformation):

    def __init__(self):
        self.poses = {}

    def write(self, dest_path):
        dest_path = Path(dest_path)
        with open(dest_path, "w") as alignment_file:
            for p in self.poses.values():
                alignment_file.write("{} {} {} {}\n".format(p["image"], p["x"], p["y"], p["z"]))

    def get_std_name(self):
        return "colmap_alignment_ecef.txt"


class ImageGeoReferenceInformation(GeoReferenceInformation):
    pass


# Register classes
Aerial2PdsmEncoder.Register(GeoReferenceInformation)
Aerial2PdsmEncoder.Register(ImageGeoReferenceInformation)
Aerial2PdsmEncoder.Register(CameraPoseGeoReferenceInformation)


class GeoAlignment:

    @staticmethod
    def ExtractAuxData(folder_dir, extension=".[tT][xX][tT]"):
        folder_dir = Path(folder_dir)
        of_a_kind = list(folder_dir.rglob("*" + extension))
        of_jpg = dict((img.with_suffix(""), img) for img in list(folder_dir.rglob("*.[jJ][pP][gG]")))
        of_png = dict((img.with_suffix(""), img) for img in list(folder_dir.rglob("*.[pP][nN][gG]")))
        img_paths = {**of_jpg, **of_png}

        mapping = {}
        file: Path
        for file in of_a_kind:
            base_name = file.with_suffix("")
            if base_name in img_paths:
                mapping[img_paths[base_name]] = file

        return mapping

    @staticmethod
    def CreateGeoHash(point):
        bits = np.unpackbits(point.astype(np.int32).view(np.uint8))
        geo_hash = [None] * len(bits)
        for i in range(0, 32):
            p = i * 3
            geo_hash[p] = bits[i]
            geo_hash[p + 1] = bits[i + 32]
            geo_hash[p + 2] = bits[i + 64]
        return np.packbits(geo_hash[:64])

    @staticmethod
    def EstimatedPoses2Colmap(database_path: Path, estimated_camera_poses):
        raise NotImplementedError("Sorry Quaternion is not know @ this point in time")
    #     conn = COLMAPDatabase.connect(database_path)
    #     images_rows = conn.execute("SELECT * FROM images").fetchall()
    #     cur = conn.cursor()
    #
    #     for row in images_rows:
    #         if row[1] in estimated_camera_poses:
    #             cur.execute("UPDATE licontacts310317 SET liemail=%s, limobile=%s WHERE % s =?" % (liemailval, limobileval, id), (constrain,))
    #
    #
    # S
    #     images_rows = db.execute("SELECT * FROM images").fetchall()
    #     cameras_rows = db.execute("SELECT * FROM cameras").fetchall()
    #     db.close()
    #
    #     images = {}
    #     for row in images_rows:
    #         old = rehash[row[1]]
    #         images[row[0]] = BaseImage(row[0], old.qvec, old.tvec, row[2], old.name, [], [])
    #     model.images = images
    #
    #     cameras = {}
    #     for row in cameras_rows:
    #         camera_id, cam_model, width, height, params, prior = row
    #         params = blob_to_array(params, np.float64)
    #         cameras[row[0]] = Camera(camera_id, CAMERA_MODEL_IDS[cam_model].model_name, width, height, params)
    #     model.cameras = cameras

    # return model


if __name__ == "__main__":
    proj = config.SparseModel(config.project_path, model_path=config.project_path + "/sparse/0")
    GeoAlignment.Dlr2Colmap(proj.images_path, proj.base_path, ECEF=False)
