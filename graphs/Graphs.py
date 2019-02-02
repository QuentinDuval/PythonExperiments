from collections import *
from dataclasses import *
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

    # TODO - could use a representation of a dict of dict to weight

    def __init__(self, vertices=[], edges=[], weights=None):
        self.adj_list = defaultdict(list)
        self.weights = weights or {}
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

    def __bool__(self):
        return len(self.adj_list) > 0

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
Find the strongly connected components of a DIRECTED graph (Tarjan's algorithm)
"""


def digraph_strongly_connected_components(graph):
    """
    Do a DFS:
    - Keep track of the lowest discovery time found in the child of each node
    - If the lowest discovery time is equal to the discovery time => root of a SCC
    - SCC will have the same lowest discovery time
    https://www.geeksforgeeks.org/tarjan-algorithm-find-strongly-connected-components/
    """
    time = 0
    discovery = {}
    lowest = {}
    current_path = set()
    visited_stack = []
    scc_list = []

    def visit(u):
        nonlocal time
        time += 1
        discovery[u] = time
        lowest[u] = time
        current_path.add(u)
        visited_stack.append(u)

        for v in graph.adjacent_vertices(u):
            if v not in discovery:
                visit(v)
                lowest[u] = min(lowest[u], lowest[v])
            elif v in current_path:
                lowest[u] = min(lowest[u], discovery[v])

        current_path.remove(u)

        # This is the root of a strongly connected component
        if lowest[u] == discovery[u]:
            scc = []
            while visited_stack and visited_stack[-1] != u:
                scc.append(visited_stack.pop())
            scc.append(visited_stack.pop())
            scc_list.append(scc)

    visit(list(graph.vertices())[0])
    return scc_list


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
    lowest = {}
    parent = {}
    result = set()

    def visit(u, source):
        nonlocal time
        time += 1
        parent[u] = source
        discovery[u] = time
        lowest[u] = time

        for v in graph.adjacent_vertices(u):
            if v not in discovery:
                visit(v, source=u)
                lowest[u] = min(lowest[u], lowest[v])
                if parent[u] and lowest[v] >= discovery[u]:
                    result.add(u)
            if v != parent[u]:
                lowest[u] = min(lowest[u], discovery[v])

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
                # Relaxation of the distance
                if heap.get_priority(e.destination) > e.weight:
                    heap.update(e.destination, e.weight)
                    parents[e.destination] = e
    return parents.values()


"""
Dijkstra's algorithm: Shortest Path with positive weights
- Observe the similitude with Prim's algorithm (only the weighting policy changes)
"""


class ShortestPathsFrom:
    def __init__(self, source, parents):
        self.source = source
        self.parents = parents

    def shortest_path_to(self, destination):
        # Observe the generator's elegance: no need to duplicate code between shortest_path_to and shortest_distance_to
        while destination != self.source:
            e = self.parents.get(destination)
            if e is None:
                yield WeightedEdge(source=None, destination=None, weight=float('inf'))
                return
            else:
                yield e
                destination = e.source

    def shortest_distance_to(self, destination):
        return sum(e.weight for e in self.shortest_path_to(destination))


def dijkstra(graph: AdjListGraph, source: any) -> ShortestPathsFrom:
    if not graph:
        return None

    heap = IndexHeap()
    for v in graph.vertices():
        heap.add(v, float('inf'))
    heap.update(source, 0)

    parents = {}
    while len(heap) > 0:
        u, distance_u = heap.pop_min()
        for e in graph.edges_from(u):
            if e.destination in heap:
                # Relaxation of the distance
                if heap.get_priority(e.destination) > e.weight + distance_u:
                    heap.update(e.destination, e.weight + distance_u)
                    parents[e.destination] = e
    return ShortestPathsFrom(source, parents)


"""
Bellman Ford's algorithm: Shortest Path with any weights
- Consider path of increasing length
- If a path is bigger than the number of vertices - 1, then there is a negative cycle
"""


class NegativeCycleDetected(Exception):
    """ Raised when a negative cycle is detected (no shortest path possible) """


def bellman_ford(graph: AdjListGraph, source: any) -> ShortestPathsFrom:
    if not graph:
        return None

    distance = {v: float('inf') for v in graph.vertices()}
    parents = {v: None for v in graph.vertices()}
    distance[source] = 0

    for _ in range(len(graph)):
        at_least_one_update = False
        for e in graph.edges():
            # Relaxation of the distance
            if distance[e.destination] > e.weight + distance[e.source]:
                distance[e.destination] = e.weight + distance[e.source]
                parents[e.destination] = e
                at_least_one_update = True
        if not at_least_one_update:
            break
    else:
        # The else in Python is triggered if the loop is finished
        for e in graph.edges():
            if distance[e.destination] > e.weight + distance[e.source]:
                raise NegativeCycleDetected()

    return ShortestPathsFrom(source, parents)

