from collections import *
from dataclasses import *
import matplotlib.pyplot as plot
import networkx as nx
from typing import List
from graphs.DisjointSets import *
from graphs.IndexHeap import *


@dataclass(repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
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
        for e in edges:
            if isinstance(e, WeightedEdge):
                self.adj_list[e.source].append(e.destination)
                self.adj_list[e.destination].append(e.source)
                self.weights[(e.source, e.destination)] = e.weight
                self.weights[(e.destination, e.source)] = e.weight
            else:
                u, v = e
                self.adj_list[u].append(v)
                self.adj_list[v].append(u)

    def __len__(self):
        return len(self.adj_list)

    def __getitem__(self, vertex):
        return self.adj_list[vertex]

    def add(self, e: WeightedEdge):
        self.adj_list[e.source].append(e.destination)
        self.adj_list[e.destination].append(e.source)
        self.weights[(e.source, e.destination)] = e.weight
        self.weights[(e.destination, e.source)] = e.weight

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
        for source in self.vertices():
            yield from self.edges_from(source)


"""
Kruskal algorithm: Minimum Spanning Tree
"""


def kruskal(graph: AdjListGraph) -> List[WeightedEdge]:
    if len(graph) == 0:
        return []

    minimum_spanning_tree = []

    edges = list(graph.edges())
    edges.sort(key=lambda e: e.weight)
    disjoint_sets = DisjointSets(graph.vertices())

    for edge in edges:
        if not disjoint_sets.joined(edge.source, edge.destination):
            disjoint_sets.union(edge.source, edge.destination)
            minimum_spanning_tree.append(edge)
            if len(minimum_spanning_tree) == len(graph) - 1:
                break

    return minimum_spanning_tree


"""
Prim's algorithm: Minimum Spanning Tree
"""


def prims(graph: AdjListGraph) -> List[WeightedEdge]:
    if len(graph) == 0:
        return []

    heap = IndexHeap()
    vertices = list(graph.vertices())
    heap.add(vertices[0], 0)
    parents = {}
    visited = set()

    while len(heap) > 0:
        u, _ = heap.pop_min()
        visited.add(u)
        for e in graph.edges_from(u):
            if e.destination in visited:
                continue
            if e.destination in heap:
                if heap.get_priority(e.destination) > e.weight:
                    heap.update(e.destination, e.weight)
                    parents[e.destination] = e
            else:
                heap.add(e.destination, e.weight)
                parents[e.destination] = e

    return parents.values()


def dijkstra():
    pass


def articulation_points():
    pass

