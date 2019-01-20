from graphs.Graphs import *

import networkx as nx
import matplotlib.pyplot as plot


def show_weighted_graph(graph: AdjListGraph):
    edges = list(graph.edges())
    graph = nx.Graph((e.source, e.destination) for e in edges)
    for e in edges:
        graph[e.source][e.destination]['weight'] = e.weight
    g_layout = nx.spring_layout(graph)
    nx.draw(graph, pos=g_layout, with_labels=True)
    nx.draw_networkx_edge_labels(graph, pos=g_layout, labels=nx.get_edge_attributes(graph, 'weight'))
    plot.show()
