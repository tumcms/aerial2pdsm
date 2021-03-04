from pathlib import Path
import networkx as nx
from numpy import cbrt
from colmap_scripts import export_inlier_matches as read_inlier_matches
from plots.plot_ego_graph import plot_ego_graph


def CoveredAreaGraph(area_name, graph_name, images={}):
    root_name = area_name
    G = nx.Graph()
    G.add_node(root_name)
    G.name = graph_name
    for name, _thingies in images.items():
        contr_features = _thingies["val"]
        G.add_node(name, full_name=_thingies["full_path"], image_id=_thingies["id"])
        G.add_edge(root_name, name, weight=contr_features)  # activate for plotting / max_features)  #
    return G


def WriteEgoGraph(G, root, path):
    path = Path(path).with_suffix(".svg")
    plot_ego_graph(G, root, True, path)


def CreateMatchingGraph(db_path):
    # images = rm.read_images_binary(model_path + "/images.bin")
    matches = read_inlier_matches.read_matches(db_path, "", 50)
    max_matches = max([i.matches for i in matches])

    G = nx.Graph()
    for match in matches:
        m1 = match.image1  # .split("/")[1]
        m2 = match.image2  # .split("/")[1]
        G.add_node(m1)
        G.add_node(m2)
        G.add_edge(m1, m2, weight=cbrt(match.matches / max_matches))  #
    return G


def CreateGraphFromDict(dict):
    G = nx.Graph()
    for i, v in dict.items():
        G.add_node()

if __name__ == "__main__":
    search_img = "------Path------"
    proj = config.SparseModel(config.project_path, db_path=config.project_path + "/katrin1.db")
    G = CreateMatchingGraph(proj.database_path)
    plot_ego_graph(G, search_img, True)
    # ego = nx.ego_graph(G, search_img, radius=1)
    # plot_circular(ego)
    # plt.show()
