from pathlib import Path

from numpy import sqrt, cbrt

import plotly.graph_objects as go
import networkx as nx
from matplotlib import pyplot as plt

import config
from colmap_scripts import read_write_model as rm
from config import project_path
from colmap_scripts import export_inlier_matches as read_inlier_matches
import plotly.graph_objects as go

search_img = "R1022.jpg"
database_path = project_path + "/katrin1.db"


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


def plot_ego_graph(graph, search, plotly=False, path="fig_graph.svg"):
    # node_and_degree = graph.degree()
    hub = search
    # degree = graph.degree[search]
    # Create ego graph of main hub
    hub_ego = nx.ego_graph(graph, hub)
    pos = nx.spring_layout(hub_ego, k=0.7, scale=0.1)

    if plotly:
        Xv = [p[0] for p in pos.values()]
        Yv = [p[1] for p in pos.values()]
        Xed = []
        Yed = []
        for edge in hub_ego.edges:
            Xed += [pos[edge[0]][0], pos[edge[1]][0], None]
            Yed += [pos[edge[0]][1], pos[edge[1]][1], None]

        trace3 = go.Scatter(x=Xed,
                            y=Yed,
                            mode='lines',
                            line=dict(color='rgb(210,210,210)', width=1),
                            hoverinfo='none'
                            )

        a = pos[hub]

        cs_trace = go.Scatter(x=[pos[hub][0]],
                              y=[pos[hub][1]],
                              mode='markers+text',
                              hoverinfo='text',
                              textposition='bottom center',
                              textfont=dict(
                                  # family="sans serif",
                                  size=13,
                                  # color="LightSeaGreen"
                              ),
                              marker=dict(
                                  color='Black',
                                  size=23,
                                  line=dict(
                                      color='Gray',
                                      width=2
                                  )
                              ),
                              showlegend=False
                              )
        node_trace = go.Scatter(x=Xv,
                                y=Yv,
                                mode='markers+text',
                                hoverinfo='text',
                                textposition='bottom center',
                                textfont=dict(
                                    # family="sans serif",
                                    size=13,
                                    # color="LightSeaGreen"
                                ),
                                marker=dict(
                                    showscale=True,
                                    # colorscale options
                                    # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                                    # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                                    # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                                    colorscale='Portland',  # Viridis',
                                    # reversescale=True,
                                    color=[],
                                    size=20,
                                    colorbar=dict(
                                        thickness=15,
                                        title='<b>Coverage</b>',
                                        xanchor='left',
                                        titleside='right',
                                        tickvals=[.1, .2, .4, .6, .8, 1],
                                        ticktext=[' 10%', ' 20%', ' 40%', ' 60%', ' 80%', '100%'],
                                        titlefont=dict(size=16),
                                        tickfont=dict(size=13)
                                    ),
                                    line_width=2))

        node_accu_edges_weight = []
        node_text = []
        for node in hub_ego.nodes():
            if node is hub:
                node_accu_edges_weight.append(0)
                node_text.append("")
                continue
                #
            else:
                acc_weight = 0
                for edge in graph.edges(node, data=True):
                    acc_weight += edge[2]["weight"]
                    break
                node_accu_edges_weight.append(acc_weight)
                node_text.append(node)

        node_trace.text = node_text
        node_trace.marker.color = node_accu_edges_weight
        cs_trace.text = hub

        # annot = "Correlating Images, distance correlates to coverage." #+ \
        # "<a href='http://nbviewer.ipython.org/gist/empet/07ea33b2e4e0b84193bd'> [2]</a>"

        data1 = [trace3, node_trace, cs_trace]
        fig1 = go.Figure(data=data1, layout=go.Layout(
            titlefont_size=14,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            # paper_bgcolor='rgba(0,0,0,0)',
            # plot_bgcolor='rgba(0,0,0,0)',
            # annotations=[dict(
            #     text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
            #     showarrow=False,
            #     xref="paper", yref="paper",
            #     x=0.005, y=-0.002)],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                         )
        # fig1['layout']['annotations'][0]['text'] = annot
        if path:
            fig1.write_image(str(path))
        else:
            fig1.show()

    else:
        nx.draw(hub_ego, pos, node_color='b', node_size=50, with_labels=True)
        # Draw ego as large and red
        nx.draw_networkx_nodes(hub_ego, pos, nodelist=[hub], node_size=300, node_color='r')


# def plot_circular(graph):
#     import matplotlib.pyplot as plt
#     import networkx as nx
#
#     try:
#         import pygraphviz
#         from networkx.drawing.nx_agraph import graphviz_layout
#     except ImportError:
#         try:
#             import pydot
#             from networkx.drawing.nx_pydot import graphviz_layout
#         except ImportError:
#             raise ImportError("This example needs Graphviz and either "
#                               "PyGraphviz or pydot")
#     # G = nx.balanced_tree(3, 5)
#     pos = graphviz_layout(graph, prog='twopi', args='')
#     plt.figure(figsize=(8, 8))
#     nx.draw(graph, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)
#     plt.axis('equal')


if __name__ == "__main__":
    proj = config.SparseModel(config.project_path, db_path=config.project_path + "/katrin1.db")
    G = CreateMatchingGraph(proj.database_path)
    plot_ego_graph(G, search_img, True)
    # ego = nx.ego_graph(G, search_img, radius=1)
    # plot_circular(ego)
    # plt.show()
