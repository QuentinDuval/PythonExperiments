from collections import *
from dataclasses import *
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


# test_graph()


class DisjointSets:
    """
    Data structure to implement Union-Find
    """

    def __init__(self, values):
        self.parents = list(range(len(values)))
        self.value_to_set = {v: i for i, v in enumerate(values)}

    def union(self, u, v):
        su = self.find(u)
        sv = self.find(v)
        if su != sv:
            self.parents[su] = sv   # TODO - Union by rank

    def find(self, u):
        s = self.value_to_set[u]
        while self.parents[self.parents[s]] != self.parents[s]:
            self.parents[s] = self.parents[self.parents[s]]
        return self.parents[s]

    def joined(self, u, v):
        return self.find(u) == self.find(v)

    def __repr__(self):
        return 'DisjointSets' + repr({
            'parents': self.parents,
            'value_to_set': self.value_to_set
        })


def test_disjoint_set():
    disjoint_set = DisjointSets(range(10))
    assert not disjoint_set.joined(1, 2)
    disjoint_set.union(1, 2)
    assert disjoint_set.joined(1, 2)
    for i in range(1, 10):
        disjoint_set.union(i-1, i)
    for i in range(1, 10):
        assert disjoint_set.joined(i-1, i)
    print(disjoint_set)


# test_disjoint_set()


def kruskal(graph: AdjListGraph) -> 'list of WeightedEdge':
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


def test_kruskal():
    # Example graph of https://en.wikipedia.org/wiki/Kruskal%27s_algorithm
    # TODO - find a way to print the graph nicely
    vertices = list('abcdefg')
    edges = [
        WeightedEdge('a', 'b', 7),
        WeightedEdge('a', 'd', 5),
        WeightedEdge('b', 'c', 8),
        WeightedEdge('b', 'd', 9),
        WeightedEdge('b', 'e', 7),
        WeightedEdge('c', 'e', 5),
        WeightedEdge('d', 'e', 15),
        WeightedEdge('d', 'f', 6),
        WeightedEdge('e', 'f', 8),
        WeightedEdge('e', 'g', 9),
        WeightedEdge('f', 'g', 11)
    ]
    graph = AdjListGraph(vertices=vertices, edges=edges)
    for e in kruskal(graph):
        print(e)


# test_kruskal()


def prims():
    pass


def dijkstra():
    pass


def articulation_points():
    pass

