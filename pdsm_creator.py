from os import path, mkdir
from subprocess import call
import numpy as np
from scipy.spatial.ckdtree import cKDTree

from colmap_scripts import read_write_model as rwm
from colmap_scripts.database import COLMAPDatabase, blob_to_array
from colmap_scripts.read_write_model import BaseImage, Camera, CAMERA_MODELS, CAMERA_MODEL_IDS, Point3D, qvec2rotmat
from config import SparseModel, project_path, GetInlineMatches, SfmModel


# due to id issues its easy to start the process partially again. Also we can increase the number of matches!
def ExtractSubModel(model: SfmModel, features_in_site, new_path):
    new_images = {}
    new_points = {}
    new_cameras = {}

    img_mapping = {}
    point_mapping = {}
    camera_mapping = {}
    used_image_ids = sorted(list(dict.fromkeys([img_id for feature in features_in_site for img_id in feature.image_ids])))
    for new_id, uid in enumerate(used_image_ids):
        img_mapping[uid] = new_id + 1

    pid_cnt = 1
    for uid in used_image_ids:
        _, pids = GetInlineMatches(model.images[uid])
        for pid in pids:
            point: Point3D = model.points[pid]

            id_list = [img_ids for img_ids in point.image_ids if img_ids in img_mapping.keys()]

            # point that is used in more than 2 images => new id, add to mapping
            if len(id_list) >= 2:
                point_mapping[pid] = pid_cnt
                new_img_ids = [img_mapping[img_id] for img_id in id_list]
                new_points[pid_cnt] = Point3D(pid_cnt, point.xyz, point.rgb, point.error, new_img_ids, point.point2D_idxs)
                pid_cnt += 1

    cid_cnt = 0
    for uid in used_image_ids:
        full_img: BaseImage = model.images[uid]
        if full_img.camera_id not in camera_mapping.keys():
            camera_mapping[full_img.camera_id] = cid_cnt
            cid_cnt += 1

        points_assosiated = [point_mapping[pid] for pid in full_img.point3D_ids if pid in point_mapping.keys()]
        new_img = BaseImage(img_mapping[uid], full_img.qvec, full_img.tvec, camera_mapping[full_img.camera_id], full_img.name, full_img.xys, points_assosiated)
        new_images[img_mapping[uid]] = new_img

    cam: Camera
    for cam_id in model.cameras:
        if cam_id in camera_mapping.keys():
            cid_id = camera_mapping[cam_id]
            cam = model.cameras[cam_id]
            new_cameras[cid_id] = Camera(cid_id, cam.model, cam.width, cam.height, cam.params)

    new_model = SparseModel(new_path)
    new_model.cameras = new_cameras
    new_model.points = new_points
    new_model.images = new_images
    return new_model


def SaveModel(model: SparseModel):
    dest_path = model.base_path + "/sparse/0"
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    rwm.write_model(model.cameras, model.images, model.points, dest_path, ".txt")


if __name__ == "__main__":
    project = SparseModel(project_path)
    test_images = ["L0798"]
    ExtractSubModel()

from shutil import copyfile
import os.path
from pathlib import Path


def PortImages(model: SfmModel, images_path: Path, dest_model: SparseModel):
    rehash = {}
    for oimg in model.images.values():
        rehash[oimg.name] = oimg

    for img_id in dest_model.images.values():
        old = rehash[img_id.name]
        dest_path = Path(dest_model.images_path) / img_id.name

        if not os.path.exists(dest_path.parent):
            os.makedirs(dest_path.parent)
        copyfile(images_path / old.name, dest_path)


def SyncModelWithDatabase(model: SparseModel):
    db = COLMAPDatabase.connect(model.database_path)
    rehash = {}
    for oimg in model.images.values():
        rehash[oimg.name] = oimg

    images_rows = db.execute("SELECT * FROM images").fetchall()
    cameras_rows = db.execute("SELECT * FROM cameras").fetchall()
    db.close()

    images = {}
    for row in images_rows:
        old = rehash[row[1]]
        images[row[0]] = BaseImage(row[0], old.qvec, old.tvec, row[2], old.name, [], [])
    model.images = images

    cameras = {}
    for row in cameras_rows:
        camera_id, cam_model, width, height, params, prior = row
        params = blob_to_array(params, np.float64)
        cameras[row[0]] = Camera(camera_id, CAMERA_MODEL_IDS[cam_model].model_name, width, height, params)
    model.cameras = cameras

    return model


# def FastLocationSearch(img_name, qps: np.array, model: SparseModel) -> [QueryMatch]:
#     #img: rm.Image = next((i for i in model.images.values() if i.name == img_name), None)
#     img = next((i for i in model.images.values() if i.name == img_name), None)
#     feature2d_coords, feature2d_ids = GetInlineMatches(img)
#     fvec = np.reshape(np.asarray(feature2d_coords, dtype=np.int64), (-1, 2))
#     tree = cKDTree(fvec, leafsize=100)
#     nn_d, nn_i = tree.query(qps)
#     return [nn_i, nn_d]

def AnalyseDepth(sub_model: SparseModel):
    min_depth = 1e9
    max_depth = -1e9
    img: BaseImage

    for img in sub_model.images.values():
        cam_pos = -1 * np.dot(qvec2rotmat(img.qvec).T, img.tvec)
        for pid in img.point3D_ids:
            point = sub_model.points[pid]
            distance = np.linalg.norm(point.xyz - cam_pos)
            if min_depth > distance:
                min_depth = distance
            elif max_depth < distance:
                max_depth = distance
            #
    return min_depth, max_depth
