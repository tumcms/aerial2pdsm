# import multiprocessing
# import time

import numpy as np
from config import Keypoint
#from config import keypoint_path, SparseModel, project_path, Keypoint
# from helpers import GetClosestFeature, FastLocationSearch
# from plotter import ImagePlot, plt


def ReadKeypointFile(path):
    kpoints = []
    with open(path, "r") as file:
        for line in file.readlines():
            if line[0] == "#":
                continue
            img_name, x, y = line.strip().split(" ")
            vec = np.array([x, y], dtype=np.float64)
            kpoints.append(Keypoint(img_name, vec))
    return kpoints

#
# #__name__ = "__MapDataset__"
# if __name__ == "__main__":
#     # post_condition
#     project = SparseModel(project_path)
#     manager = multiprocessing.Manager()
#
#     # work
#     keypoints = ReadKeypointFile(keypoint_path)
#
#     for kp in keypoints:
#         matches = [GetClosestFeature(kp, project)]
#         plot = ImagePlot(project)
#         plot.PlotNext(matches)
#         plt.savefig("")
#         plt.show()
#
# if __name__ == "__MapDataset__":
#     # post_conditionx-special/nautilus-clipboard
#     # copy
#     # file:///home/felix/pointclouds/_working/2019_11_07_Muenchen_26_10_2018/sparse/0
#     project = SparseModel("/home/felix/pointclouds/_working/2019_11_12_Muenchen_04_07_2017", model_path="/home/felix/pointclouds/_working/2019_11_12_Muenchen_04_07_2017/sparse2/0")
#
#
#     # work
#     # keypoints = ReadKeypointFile(keypoint_path)
#     images = project.images
#     cameras = project.cameras
#
#     error = []
#     error_max = []
#     parts = 1
#
#     starttime = time.time()
#     tttt = 0
#
#     starttime = time.time()
#     for cnt, img in enumerate(images.values()):
#         # if cnt > 5 and img.name[0] == "r":
#         #     continue
#         # elif img.name[0] == "l":
#         #     tttt += 1
#         #     if tttt > 5:
#         #         break
#         starttime_img = time.time()
#         cam = cameras[img.camera_id]
#         # if cam.id == 2:
#         h = np.arange(cam.height, dtype=np.int64)
#         w = np.arange(cam.width, dtype=np.int64)
#     # else:
#         #     w = np.arange(cam.height, dtype=np.int64)
#         #     h = np.arange(cam.width, dtype=np.int64)
#
#         xx, yy = np.meshgrid(w, h)
#         all_pixels = np.array_split(np.vstack([xx.reshape(-1), yy.reshape(-1)]).T, parts, axis=0)
#
#         for pcnt, partial in enumerate(all_pixels):
#             id, distances = FastLocationSearch(img.name, partial, project)
#             mean = np.mean(distances)
#             max = np.max(distances)
#             print("Image {}, Mean {:.2f}, Max {:.2f} and took {}".format(img.name, mean, max, time.time() - starttime_img))
#             error.append(mean)
#             error_max.append(max)
#         # if cnt > 2:
#         #     break
#
#     tt = time.time() - starttime
#     print("The overall mean error was {:.2f} and took {:.2f}".format(np.mean(error), tt))
#     print("This means:max{} mean_max {} time_per_image {} ".format(np.max(error_max), np.mean(error_max), tt / len(images)))
#
#     images__ = list(images.values())
#     with open("error.txt", "w") as f:
#         for i in range(len(error)):
#             f.writelines("{} {:.2f} {:.2f}\n".format(images__[i].name, error[i], error_max[i]))
#
#
#     # from joblib import Parallel, delayed
#     # for kp in keypoints:
#     #     matches = [GetClosestFeature(kp, project)]
#     #     plot = ImagePlot(project)
#     #     plot.PlotNext(matches)
#     #     plt.show()
