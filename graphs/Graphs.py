from collections import *
from dataclasses import *
import matplotlib.pyplot as plot
import networkx as nx
import numpy as np
from typing import List


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


def kruskal(graph: AdjListGraph) -> List[WeightedEdge]:
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

    # Showing the graph
    graph = nx.Graph((e.source, e.destination) for e in edges)
    for e in edges:
        graph[e.source][e.destination]['weight'] = e.weight
    g_layout = nx.spring_layout(graph)
    nx.draw(graph, pos=g_layout, with_labels=True)
    nx.draw_networkx_edge_labels(graph, pos=g_layout, labels=nx.get_edge_attributes(graph, 'weight'))
    plot.show()


# test_kruskal()


"""
Binary heap which supports a update-key
"""


class IndexHeap:
    def __init__(self):
        self.index = {}
        self.values = [(None, -1 * float('inf'))]

    def __len__(self):
        return len(self.values) - 1

    def __repr__(self):
        return 'IndexHeap' + repr({
            'index': self.index,
            'values': self.values
        })

    def min(self):
        return self.values[1][0]

    def pop_min(self):
        min_key, min_prio = self.values[1]
        if len(self.values) > 2:
            key, prio = self.values.pop()
            self.values[1] = (key, prio)
            self.index[key] = 1
            self._dive(1)
        else:
            self.values.pop()
        del self.index[min_key]
        return min_key, min_prio

    def add(self, key, priority):
        self.values.append((key, priority))
        last_index = len(self.values) - 1
        self.index[key] = last_index
        self._swim(last_index)

    def update(self, key, priority):
        if key not in self.index:
            self.add(key, priority)
        else:
            idx = self.index[key]
            _, previous_priority = self.values[idx]
            self.values[idx] = key, priority
            if priority > previous_priority:
                self._dive(idx)
            else:
                self._swim(idx)

    def __contains__(self, key):
        return key in self.index

    def get_priority(self, key):
        return self.values[self.index[key]][1]

    def _swim(self, i):
        while self.values[i][1] < self.values[i//2][1]:
            self._swap(i, i//2)
            i = i // 2

    def _dive(self, i):
        while True:
            prio = self.values[i][1]
            l_prio = self.values[i*2][1] if i*2 < len(self.values) else float('inf')
            r_prio = self.values[i*2+1][1] if i*2+1 < len(self.values) else float('inf')
            if prio <= max(l_prio, r_prio):
                break

            child = i*2 if l_prio < r_prio else i*2+1
            self._swap(i, child)

    def _swap(self, i, j):
        ki = self.values[i][0]
        kj = self.values[j][0]
        self.index[ki], self.index[kj] = self.index[kj], self.index[ki]
        self.values[i], self.values[j] = self.values[j], self.values[i]


def test_index_heap():
    heap = IndexHeap()
    for c in "ghaibcjdef":
        heap.add(c, ord(c))
    assert 'a' == heap.min()
    heap.update('c', ord('a') - 1)
    assert 'c' == heap.min()
    heap.pop_min()
    assert 'a' == heap.min()


# test_index_heap()


"""
Prim's algorithm
"""


def prims(graph: AdjListGraph) -> List[WeightedEdge]:
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


def test_prims():
    # Example graph of https://en.wikipedia.org/wiki/Kruskal%27s_algorithm
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
    for e in prims(graph):
        print(e)

    # Showing the graph
    graph = nx.Graph((e.source, e.destination) for e in edges)
    for e in edges:
        graph[e.source][e.destination]['weight'] = e.weight
    g_layout = nx.spring_layout(graph)
    nx.draw(graph, pos=g_layout, with_labels=True)
    nx.draw_networkx_edge_labels(graph, pos=g_layout, labels=nx.get_edge_attributes(graph, 'weight'))
    plot.show()


# test_prims()


def dijkstra():
    pass


def articulation_points():
    pass

