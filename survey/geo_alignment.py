from pathlib import Path
import numpy as np


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


# if __name__ == "__main__":
#     proj = config.SparseModel(config.project_path, model_path=config.project_path + "/sparse/0")
#     GeoAlignment.Dlr2Colmap(proj.images_path, proj.base_path, ECEF=False)
