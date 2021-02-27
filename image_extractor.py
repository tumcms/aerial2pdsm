from closest_keypoint import ReadKeypointFile
from config import keypoint_path, SparseModel, project_path, Keypoint, QueryMatch
from helpers import GetClosestFeature
from plotter import ImagePlot, plt

if __name__ == "__main__":
    # post_condition
    project = SparseModel(project_path, model_path=project_path + r"/sparse/aligned")
    images = project.images
    if True:
        query_point = "rechts/R1022.jpg"
        img = next((i for i in images.values() if i.name == query_point), None)

    sim = sorted([i.name.split("/")[1] for i in images.values()])
    print(len(sim))
    print(sim[0], sim[-1])

    # work
    # keypoints = ReadKeypointFile(keypoint_path)
    # for kp in keypoints:
    #     matches = [GetClosestFeature(kp, project)]
    #     plot = ImagePlot(project)
    #     plot.PlotNext(matches)
    #     plt.show()
