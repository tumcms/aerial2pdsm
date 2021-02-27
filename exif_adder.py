from pathlib import Path

from piexif import GPSIFD, load, dump, insert
import config
from os import listdir, path
from fractions import Fraction

from geo_alignment import pattern_write, ReadAux_DLR


# from c060604 github
def to_deg(value, loc):
    if value < 0:
        loc_value = loc[0]
    elif value > 0:
        loc_value = loc[1]
    else:
        loc_value = ""
    abs_value = abs(value)
    deg = int(abs_value)
    t1 = (abs_value - deg) * 60
    min = int(t1)
    sec = round((t1 - min) * 60, 5)
    return deg, min, sec, loc_value


# from c060604 github
def change_to_rational(number):
    f = Fraction(str(number))
    return f.numerator, f.denominator


def Gps2Exif(image_path: Path, lon, lat, alt):
    if image_path.suffix.lower() == ".jpg":
        try:
            exif_dict = load(str(image_path))
        except:
            print(image_path)
        lat_deg = to_deg(float(lat), ["S", "N"])
        lng_deg = to_deg(float(lon), ["W", "E"])

        exiv_lat = (change_to_rational(lat_deg[0]), change_to_rational(lat_deg[1]), change_to_rational(lat_deg[2]))
        exiv_lng = (change_to_rational(lng_deg[0]), change_to_rational(lng_deg[1]), change_to_rational(lng_deg[2]))

        gps_ifd = {
            GPSIFD.GPSVersionID: (2, 0, 0, 0),
            GPSIFD.GPSAltitudeRef: 1,
            GPSIFD.GPSAltitude: change_to_rational(round(float(alt))),
            GPSIFD.GPSLatitudeRef: lat_deg[3],
            GPSIFD.GPSLatitude: exiv_lat,
            GPSIFD.GPSLongitudeRef: lng_deg[3],
            GPSIFD.GPSLongitude: exiv_lng,
        }

        exif_dict["GPS"] = gps_ifd
        exif_bytes = dump(exif_dict)
        insert(exif_bytes, str(image_path))


def RewriteAuxExif(image_folder_path):
    for ipath in listdir(image_folder_path):
        img_path = path.join(image_folder_path, ipath)
        if img_path.endswith(("jpg", "jpeg")):
            img_dat = ReadAux_DLR(img_path)
            lon, lat, alt = pattern_write.findall(img_dat["GPSIMU"]["Position"])[0]
            Gps2Exif(img_path, lon, lat, alt)


if __name__ == "__main__":
    base_path = "/home/felix/pointclouds/_working/2019_11_12_Muenchen_2019_06_28"
    proj = config.SparseModel(config.project_path if not base_path else base_path)
    for fname in listdir(proj.images_path):
        folder = path.join(proj.images_path, fname)
        if path.isdir(folder):
            RewriteAuxExif(folder)
    print("GPS data added!")
