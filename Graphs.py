from collections import *
from dataclasses import *
from types import *
import numpy as np


@dataclass
class WeightedEdge:
    source: any
    destination: any
    weight: float = 1

    def __len__(self):
        return 3

    def __getitem__(self, index):
        """
        Make it compatible with tuple / list destructuring
        """
        if index == 0:
            return self.source
        elif index == 1:
            return self.destination
        return self.weight


class AdjListGraph:
    """
    Give weights and it becomes a weighted graph (weights are equal to 1 by default)
    """

    def __init__(self, vertices=[], edges=[], weights={}):
        self.adj_list = defaultdict(list)
        self.weights = weights
        for v in vertices:
            self.adj_list[v] = []
        for u, v in edges:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def __len__(self):
        return len(self.adj_list)

    def __getitem__(self, vertex):
        return self.adj_list[vertex]

    def adjacent_vertices(self, source):
        return self.adj_list[source]

    def edges_from(self, source):
        for destination in self.adj_list[source]:
            yield WeightedEdge(source=source,
                               destination=destination,
                               weight=self.weights.get((source, destination), 1))

    def vertices(self):
        return self.adj_list.keys()

    def edges(self):
        return self.adj_list.values()


def test_graph():
    vertices = [0, 1, 2, 3, 4, 5]
    edges = list((u, v) for u in vertices for v in vertices)
    weights = [np.random.normal(loc=5, scale=2) for _ in edges]
    graph = AdjListGraph(vertices=vertices, edges=edges, weights=weights)

    visited = set()
    parents = {}
    to_visit = [0]
    while to_visit:
        u = to_visit.pop()
        visited.add(u)
        for v in graph.adjacent_vertices(u):
            if v not in visited:
                parents[v] = u
                to_visit.append(v)
    print("DFS Tree:", parents)


def kruskal():
    pass


def prims():
    pass


def dijkstra():
    pass


def articulation_points():
    pass

