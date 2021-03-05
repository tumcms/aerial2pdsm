from survey.sfm_helpers import GetInlineMatches, QueryMatch
import matplotlib.pyplot as plt

plt.ioff()
import matplotlib.image as mpimg


def PlotNext(image_base_dir, plot, query_matches: [QueryMatch], nopoints=False, size=6, extra_size=25):
    img_bucket = {}
    for qm in query_matches:
        if qm.image.id not in img_bucket:
            img_bucket[qm.image.id] = []
        img_bucket[qm.image.id].append(qm)

    for img_id, group in img_bucket.items():
        plot.sub.append(plt.subplot())
        plot.sub[-1].title.set_text(img_id)

        abs_path = image_base_dir / group[0].image.name
        plt_image = mpimg.imread(abs_path)
        plt.imshow(plt_image)
        cur_points2d_coord, cur_points3d_ids = GetInlineMatches(group[0].image)
        if not nopoints:
            plt.scatter(cur_points2d_coord[:, 0], cur_points2d_coord[:, 1], s=size, facecolors='firebrick', edgecolors='darkred')
            for qmp in group:
                plt.scatter(qmp.query_point2d[0], qmp.query_point2d[1], s=extra_size, edgecolors='blue')
                plt.scatter(qmp.point2d_coord[0], qmp.point2d_coord[1], s=extra_size, facecolors='g', edgecolors='g')
                # print(ecef2geodetic(qmp.point3d.xyz[0], qmp.point3d.xyz[1], qmp.point3d.xyz[2]))
