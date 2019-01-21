from collections import *
from dataclasses import *
import matplotlib.pyplot as plot
import networkx as nx
from typing import List
from graphs.DisjointSets import *
from graphs.IndexHeap import *
import heapq


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
                self.add(e)
            else:
                u, v = e
                self.adj_list[u].append(v)
                self.adj_list[v].append(u)

    def __repr__(self):
        return 'AdjListGraph ' + repr({
            'adj_list': self.adj_list,
            'weights': self.weights
        })

    def __len__(self):
        return len(self.adj_list)

    def __getitem__(self, vertex):
        return self.adj_list[vertex]

    def add(self, e: WeightedEdge):
        self.add_directed(e)
        self.add_directed(WeightedEdge(source=e.destination, destination=e.source, weight=e.weight))

    def add_directed(self, e: WeightedEdge):
        self.adj_list[e.source].append(e.destination)
        self.weights[(e.source, e.destination)] = e.weight

    def adjacent_vertices(self, source):
        return self.adj_list.get(source, [])

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
Find the articulation points of a graph (Tarjan's algorithm)
"""


def articulation_points(graph):
    """
    Do a DFS:
    - Keep track of the lowest discovery time found in the child of each node
    - If the lowest discovery time is after the time of the node => articulation point
    https://www.geeksforgeeks.org/articulation-points-or-cut-vertices-in-a-graph/
    """
    time = 0
    discovery = {}
    child_lowest_discovery = {}
    parent = {}
    result = set()

    def visit(u, source):
        nonlocal time
        time += 1
        parent[u] = source
        discovery[u] = time
        child_lowest_discovery[u] = time

        for v in graph.adjacent_vertices(u):
            if v not in discovery:
                visit(v, source=u)
                if parent[u] and child_lowest_discovery[v] >= discovery[u]:
                    result.add(u)
            if v != parent[u]:
                child_lowest_discovery[u] = min(child_lowest_discovery[u], discovery[v])

    start_vertex = list(graph.vertices())[0]
    visit(start_vertex, None)

    if len([u for u, p in parent.items() if p == start_vertex]) > 1:
        result.add(start_vertex)
    return result


"""
Topological sorting
"""


class CycleDetected(Exception):
    """ Raised when a graph is not a DAG as expected """
    def __init__(self, graph, u, v):
        self.graph = graph
        self.source = u
        self.destination = v


def topological_sort(graph: AdjListGraph) -> List[any]:
    result = []
    visited = set()
    current_path = set()

    def visit(u):
        current_path.add(u)
        visited.add(u)
        for v in graph.adjacent_vertices(u):
            if v in current_path:
                raise CycleDetected(graph, u, v)
            if v not in visited:
                visit(v)
        result.append(u)
        current_path.remove(u)

    for v in graph.vertices():
        if v not in visited:
            visit(v)
    return reversed(result)


def topological_sort_2(graph: AdjListGraph) -> List[any]:
    time = 0
    start_visit = {}
    end_visit = {}

    def visit(u):
        nonlocal time
        time += 1
        start_visit[u] = time
        for v in graph.adjacent_vertices(u):
            if v not in start_visit:
                visit(v)
            elif v not in end_visit:
                raise CycleDetected(graph, u, v)
        time += 1
        end_visit[u] = time

    for v in graph.vertices():
        if v not in start_visit:
            visit(v)
    return [node for node, time in sorted(end_visit.items(), key=lambda p: p[1], reverse=True)]


"""
Kruskal algorithm: Minimum Spanning Tree
"""


def kruskal(graph: AdjListGraph) -> List[WeightedEdge]:
    if len(graph) == 0:
        return []

    edges = list(graph.edges())
    edges.sort(key=lambda e: e.weight)
    disjoint_sets = DisjointSets(graph.vertices())

    minimum_spanning_tree = []
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


def prims_slow(graph: AdjListGraph) -> List[WeightedEdge]:
    if len(graph) == 0:
        return []

    minimum_spanning_tree = []
    start_vertex = list(graph.vertices())[0]
    visited = {start_vertex}

    edge_heap = [(e.weight, e) for e in graph.edges_from(start_vertex)]
    heapq.heapify(edge_heap)

    while edge_heap:
        _, e = heapq.heappop(edge_heap)
        if e.destination in visited:
            continue

        visited.add(e.destination)
        minimum_spanning_tree.append(e)
        for e in graph.edges_from(e.destination):
            if e.destination not in visited:
                heapq.heappush(edge_heap, (e.weight, e))

    return minimum_spanning_tree


def prims(graph: AdjListGraph) -> List[WeightedEdge]:
    if len(graph) == 0:
        return []

    heap = IndexHeap()
    vertices = list(graph.vertices())

    heap.add(vertices[0], 0)
    for v in vertices[1:]:
        heap.add(v, float('inf'))

    parents = {}
    while len(heap) > 0:
        u, _ = heap.pop_min()
        for e in graph.edges_from(u):
            if e.destination in heap:
                if heap.get_priority(e.destination) > e.weight:
                    heap.update(e.destination, e.weight)
                    parents[e.destination] = e
    return parents.values()


"""
Dijkstra's algorithm: Shortest Path with positive weights
"""


def dijkstra():
    pass

