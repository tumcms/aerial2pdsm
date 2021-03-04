from pathlib import Path
import numpy as np
import pymap3d as pm
from data_io.encoder_init import Aerial2PdsmEncoder
from survey.geo_alignment import GeoAlignment
from data_io.dlr_reader import ReadAux_DLR, pattern_write
CameraPose = np.dtype([('image', "U32"), ('x', np.float64), ('y', np.float64), ('z', np.float64)])


# e.g. GPS during recording
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
