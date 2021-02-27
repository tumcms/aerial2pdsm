# from pathlib import Path
#
# import matplotlib.patches as patches
# import matplotlib.pyplot as plt
# from colmap_scripts.read_write_model import Point3D
# import pymap3d as pm
# from closest_keypoint import ReadKeypointFile
# from config import keypoint_path, project_path, SparseModel, Keypoint, QueryMatch
# from helpers import GetClosestFeature
# from plotter import ImagePlot
# import numpy as np
# from numpy.linalg import inv
# from graph_plot import nx as graph_nx, plot_ego_graph
# import plyfile
#
# def PrintAxis():
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#     import numpy as np
#     soa = np.array([[0, 0, 0, x_axis[0], x_axis[1], x_axis[2]],
#                     [0, 0, 0, y_axis[0], y_axis[1], y_axis[2]],
#                     [0, 0, 0, z_axis[0], z_axis[1], z_axis[2]]
#                     ])
#     X, Y, Z, U, V, W = zip(*soa)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.quiver(X, Y, Z, U, V, W)
#     ax.set_xlim([-1, 2])
#     ax.set_ylim([-1, 2])
#     ax.set_zlim([-1, 2])
#     plt.show()
#     return np
#
#
# # def PointsToPly(pcloud: [Point3D], filename="some_file.ply"):
# #     # my, mz, m00.point3d.xyz, m11.point3d.xyz, site3d
# #     try:
# #         vertex = np.array([i.xyz for i in pcloud])  # np.stack([mx, my, mz, m00.point3d.xyz])
# #     except:
# #         vertex = np.array([i for i in pcloud])
# #
# #     vertex.dtype = [("x", "f8"), ("y", "f8"), ("z", "f8")]
# #     vertex = vertex.squeeze(axis=1)
# #     edges = np.array([
# #         ([0, 3], 255, 0, 0),
# #         ([0, 3], 0, 255, 0),
# #         ([0, 3], 255, 0, 255)],
# #         dtype=[("index", "i4", (2,)),
# #                ("red", "u1"),
# #                ("green", "u1"),
# #                ("blue", "u1")])
# #     v = plyfile.PlyElement.describe(vertex, "vertices")
# #     e = plyfile.PlyElement.describe(edges, "edges")
# #     plyfile.PlyData([v], text=True).write(filename)
#
#
#
#
#
# def CreateLine(p1, p2, path_to_plyfile):
#     diag_interp_x = np.linspace(p1[0], p2[0], 100)
#     diag_interp_y = np.linspace(p1[1], p2[1], 100)
#     diag_interp_z = np.linspace(p1[2], p2[2], 100)
#     diag3d_points = []
#     diag3d_points = zip(diag_interp_x, diag_interp_y, diag_interp_z)
#     PointsToPly(diag3d_points, path_to_plyfile)
#
#
# debug_axis = False
# save_graph = False
# show_graph = False
# save_image = False
# show_image = False
# create_graph = True
# save_sparse_pdsm = False
# save_pdsm_info = False
# extract_binary_model = True
# save_transform = True
#
#
# def InvertTransformation(trafo_matrix):
#     inv = np.zeros((4, 4))
#     inv[:3, :3] = trafo_matrix[:3, :3].T
#     inv[:3, 3] = - np.dot(inv[:3, :3], trafo_matrix[:3, 3])
#     inv[3, 3] = 1
#     return inv
#
#
# if __name__ == "__main__":
#     # pre_condition
#     from os import path, mkdir
#
#     project = SparseModel(project_path)
#     project.parse_binary_model()
#     keypoints = ReadKeypointFile(keypoint_path)
#     compare_graphs = []
#     image_pair = 0
#
#     # work
#     for c00, c11 in pairwise(keypoints):
#         print("SiteSearch: Looking for x:({},{}) : y:({},{}) in {}".format(c00.point[0], c00.point[1], c11.point[0], c11.point[1], c00.image))
#         folder_name = "{1}_{0}".format(c00.image.split("/")[1][:-4], image_pair)
#         folder_path = path.join(project.dat_path, folder_name)
#
#         m00 = GetClosestFeature(c00, project)
#         if not m00.valid: continue
#         m11 = GetClosestFeature(c11, project)
#         if not m11.valid: continue
#
#         if not path.exists(folder_path):
#             mkdir(folder_path)
#
#         if show_image:
#             plot = ImagePlot(project)
#             plot.PlotNext([m00, m11], nopoints=False)
#             diff = m11.point2d_coord - m00.point2d_coord
#             rect = patches.Rectangle((m00.point2d_coord[0], m00.point2d_coord[1]), diff[0], diff[1], linewidth=2, edgecolor='c', facecolor='none')
#             plot.sub[-1].add_patch(rect)
#             plt.show()
#             del plot
#         if save_image:
#             plot = ImagePlot(project)
#             plot.PlotNext([m00, m11], nopoints=False, size=0.5, extra_size=2)
#             diff = m11.point2d_coord - m00.point2d_coord
#             rect = patches.Rectangle((m00.point2d_coord[0], m00.point2d_coord[1]), diff[0], diff[1], linewidth=1, edgecolor='c', facecolor='none')
#             plot.sub[-1].add_patch(rect)
#             plt.savefig(folder_path + "/{1}_{0}.svg".format(c00.image.split("/")[1][:-4], image_pair), dpi=400)
#
#         # Get Z Axis with the normal of the WGS-84
#         wgs_m00 = pm.ecef2geodetic(m00.point3d.xyz[0], m00.point3d.xyz[1], m00.point3d.xyz[2])
#         zdiff = wgs_m00[2] - 50
#         mxx = pm.geodetic2ecef(wgs_m00[0], wgs_m00[1], zdiff)
#         z_axis = Unit(np.array([m00.point3d.xyz[0] - mxx[0], m00.point3d.xyz[1] - mxx[1], m00.point3d.xyz[2] - mxx[2]]))
#
#         # create zaxis for debug
#         if debug_axis:
#             CreateLine(m00.point3d.xyz, mxx, folder_path + "/{1}_{0}_zaxis.ply".format(m00.image.name.split("/")[1][:-4], image_pair))
#
#         # project the vector from point m00 -> m11 to the plane created by the z-vector
#         hyp_3d = Project2Plane(m11.point3d.xyz - m00.point3d.xyz, z_axis)
#         # create diagonal for debug
#         if debug_axis:
#             CreateLine(m00.point3d.xyz, m00.point3d.xyz + hyp_3d, folder_path + "/{1}_{0}_diagonal3D.ply".format(m00.image.name.split("/")[1][:-4], image_pair))
#
#         # Get X Axis by using the angle between 2d X & hypotenuse in z-plane
#         c10 = Keypoint(c11.image, np.array([c11.point[0], c00.point[1]]))  # image space, right point
#         hyp_2d = c11.point - c00.point  # image space, diagonal
#         ref_2d = c10.point - c00.point  # image space, x-axis
#         theta = -1 * np.arccos(np.dot(ref_2d, hyp_2d) / (np.linalg.norm(ref_2d) * np.linalg.norm(hyp_2d)))  # angle hyp and ref
#
#         # Create Pointed Lines for visualisation
#         x_axis = Unit(RotateAroundAxis(hyp_3d, z_axis))
#         y_axis = Unit(np.cross(z_axis, x_axis))
#         if debug_axis:
#             CreateLine(m00.point3d.xyz, m00.point3d.xyz + x_axis * 50, folder_path + "/{1}_{0}_xaxis.ply".format(m00.image.name.split("/")[1][:-4], image_pair))
#             CreateLine(m00.point3d.xyz, m00.point3d.xyz + y_axis * 50, folder_path + "/{1}_{0}_yaxis.ply".format(m00.image.name.split("/")[1][:-4], image_pair))
#
#         image_base = np.column_stack([x_axis, y_axis, z_axis])
#
#         trafo_matrix = np.zeros((4, 4))
#         trafo_matrix[:3, :3] = image_base
#         trafo_matrix[:3, 3] = np.array([-m00.point3d.xyz])
#         trafo_matrix[3, 3] = 1
#
#         inverse_transformation = InvertTransformation(trafo_matrix)
#         np.savetxt(folder_path + "/{1}_{0}_2global.out".format(c00.image.split("/")[1][:-4], image_pair), inverse_transformation, delimiter="\t")
#
#
#         nb00 = np.dot(image_base.T, m00.point3d.xyz)
#         nb11 = np.dot(image_base.T, m11.point3d.xyz)
#
#         # if False: PrintAxis()
#
#         minx = min(nb00[0], nb11[0])
#         miny = min(nb00[1], nb11[1])
#         maxx = max(nb00[0], nb11[0])
#         maxy = max(nb00[1], nb11[1])
#
#         # isolate all points that belong to construction site, regardless of the image. IMPORTANT not images but points on the construction site
#         # THIS should be redone soon, its ugly and error prone
#         points_within_site = SitePoints3D(project, [minx, maxx, miny, maxy], image_base)
#         points_in_rect, max_features = CountMatches(project, points_within_site)
#         print("Points on CS: {0}".format(len(points_in_rect)))
#
#         # Create Graph
#         if create_graph:
#             root_name = "ConstructionSite"
#
#             G = graph_nx.Graph()
#             G.add_node(root_name)
#             G.name = m00.image.name.split("/")[1][:-4]
#             for name, _thingies in points_in_rect.items():
#                 contr_features = _thingies["val"]
#                 G.add_node(name, full_name=_thingies["full_path"])
#                 G.add_edge(root_name, name, weight=contr_features)  # activate for plotting / max_features)  #
#
#             print("Graph Creation")
#             # for node in G:
#             #     print(node)
#             if show_graph:
#                 plot_ego_graph(G, root_name, True)
#                 plt.show()
#             if save_graph:
#                 plot_ego_graph(G, root_name, True, folder_path + "/{1}_{0}_graph.svg".format(c00.image.split("/")[1][:-4], image_pair))
#
#             compare_graphs.append(G)
#
#         if save_sparse_pdsm:
#             PointsToPly(points_within_site, folder_path + "/{1}_{0}.ply".format(c00.image.split("/")[1][:-4], image_pair))
#
#         # The reuse of know poses with colmap will speed up the process but might intoduce scary filtering. Therefore we will use the information to
#         # reevaluate instead of working on fixed data. This varies from model, to model.
#         if extract_binary_model:
#             import pdsm_creator
#             from colmap_automate.app import Reconstructor, CreateDirectory, File2Commands, ReconstructionConfig
#
#             contributing_images = []
#             graph = compare_graphs[-1]
#
#             for root, img, weight in graph.edges(data=True):
#                 contributing_images.append(graph.nodes[img]["full_name"])
#
#             # contributing_images = [img.full_name for img in compare_graphs[eq[0].number]]
#             new_base_path = folder_path + "/colmap_model"
#
#             # Exctract a Model with images only, the ids will be continues
#             sub_model = pdsm_creator.ExtractSubModel(project, points_within_site, new_base_path)
#             min_depth, max_depth = pdsm_creator.AnalyseDepth(sub_model)
#
#             CreateDirectory(Path(sub_model.base_path))
#             # Copies needed images
#             pdsm_creator.PortImages(project, sub_model)
#             pdsm_creator.AnalyseDepth(sub_model)
#             # sets up the reconstruction
#             this_file_path = Reconstructor.GetPathOfCurrentFile()
#             source_config = this_file_path.parent / "tconfig"
#             rec_conf = ReconstructionConfig.CreateStandardConfig(
#                 sub_model.base_path, database_path=Path(sub_model.base_path, "pdsm.db"), min_depth=min_depth - 10, max_depth=max_depth)
#             rec_conf.image_global_list = Path(project.base_path, "alignment_file_gps.txt")
#             dest_config = Path(sub_model.base_path, "tconfig")
#
#             # This might look like a little freeze, but its actually taking time for the models.
#             Reconstructor.Generic2SpecificJobFiles(source_config, dest_config, rec_conf)
#             Reconstructor.execute_all(dest_config, rec_conf)
#
#             # This would sync the model to a reconstruction. This will be skipped due to colmap removing some important points
#             # sub_model = pdsm_creator.SyncModelWithDatabase(sub_model)
#             # pdsm_creator.SaveModel(sub_model)
#
#             # sub_model.database_path = new_base_path + "/" + "pdsm.db"
#
#             # if not path.exists(sub_model.base_path):
#             #    mkdir(sub_model.base_path)
#
#             # run some
#             exit()
#             pdsm_creator.ReconstructModel(sub_model, folder_path + "/dense_site.ply")
#
#         image_pair += 1
#
#
#     # identify multiple labels by graph analysis
#
#     # def diff(first, second):
#     #     second = set(second)
#     #     return [item for item in first if item not in second]
#     #
#     #
#     # for g in range(1, len(compare_graphs)):
#     #     print(diff(compare_graphs[g - 1], compare_graphs[g]))
#
#     detected_group = []
#     compare_graphs.sort(key=len, reverse=True)
#     graphnr = 0
#     for g1 in compare_graphs:
#         found = False
#         g1.number = graphnr
#         graphnr += 1
#         g1.value = len(g1)
#         g1.feature_count = sum([n[2]["weight"] for n in g1.edges(data=True)])
#         g1.mean = -1
#
#         for exist in detected_group:
#             group_similarity = []
#             for edge in g1.edges("ConstructionSite"):
#                 base_node = edge[0]
#                 far_node = edge[1]
#                 e2 = exist[0].get_edge_data(base_node, far_node)
#                 if e2:
#                     e1 = g1.get_edge_data(base_node, far_node)
#                     w1 = e1["weight"] / g1.feature_count
#                     w2 = e2["weight"] / exist[0].feature_count
#                     diff = w2 - w1
#                     ndiff = diff / w1 if diff > 0 else diff / w2
#                     group_similarity.append(abs(ndiff))
#                 else:
#                     group_similarity.append(1)
#
#             mean = sum(group_similarity) / len(group_similarity)
#             if mean < 0.4:
#                 g1.value = len(g1)
#                 exist.append(g1)
#                 exist.sort(key=len, reverse=True)
#                 found = True
#                 g1.mean = mean
#                 break
#         if not found:
#             g1.value = len(g1)
#             g1.mean = 0
#             detected_group.append([g1])
#
#     # write dependency information
#     if save_pdsm_info:
#         import xml.etree.ElementTree as ET
#
#         root = ET.Element("construction_sites")
#         for cnt, eq in enumerate(detected_group):
#             group = ET.SubElement(root, "Group_{}".format(cnt))
#             for cs in eq:
#                 csite = ET.SubElement(group, cs.name)
#                 features = ET.SubElement(csite, "features").text = str(cs.feature_count)
#                 nrcont = ET.SubElement(csite, "number_of_contributing_images").text = str(cs.value)
#                 cont_img = ET.SubElement(csite, "images")
#                 for cimg in compare_graphs[cs.number]:
#                     ET.SubElement(cont_img, cimg)
#
#         tree = ET.ElementTree(root)
#         tree.write(project.dat_path + "/equivalent_constructionsites.xml")
#
#     # for cnt, i in enumerate(detected_group):
#     #     print("Group {}:".format(cnt))
#     #     for pp in i:
#     #         print(pp.number, pp.value, pp.mean)
