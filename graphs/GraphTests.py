from graphs.DisjointSets import *
from graphs.Graphs import *
from graphs.Debugging import *

import numpy as np
import unittest

from hypothesis import *
from hypothesis.strategies import *


class TestGraph(unittest.TestCase):

    def test_graph_construction(self):
        vertices = [0, 1, 2, 3, 4, 5]
        edges = list((u, v) for u in vertices for v in vertices)
        weights = [np.random.normal(loc=5, scale=2) for _ in edges]
        graph = AdjListGraph(vertices=vertices, edges=edges, weights=weights)

        visited = set()
        parents = {0: None}
        to_visit = [0]
        while to_visit:
            u = to_visit.pop()
            visited.add(u)
            for v in graph.adjacent_vertices(u):
                if v not in visited:
                    parents[v] = u
                    to_visit.append(v)
        self.assertSetEqual(set(vertices), set(parents.keys()))

    def test_digraph_strongly_connected_components(self):
        nodes = list('abcdefghij')
        graph = AdjListGraph(vertices=nodes)

        # SCC 1
        graph.add_directed(WeightedEdge("a", "c"))
        graph.add_directed(WeightedEdge("c", "b"))
        graph.add_directed(WeightedEdge("b", "a"))
        graph.add_directed(WeightedEdge("b", "d"))

        # SCC 2
        graph.add_directed(WeightedEdge("f", "g"))
        graph.add_directed(WeightedEdge("g", "h"))
        graph.add_directed(WeightedEdge("h", "f"))

        # SCC 3
        graph.add_directed(WeightedEdge("i", "j"))
        graph.add_directed(WeightedEdge("j", "i"))

        # Connections from SCC 1 to SCC 2 and SCC 3
        graph.add_directed(WeightedEdge("d", "f"))
        graph.add_directed(WeightedEdge("d", "i"))

        # Cross-edge between SCC 2 and SCC 3
        graph.add_directed(WeightedEdge("i", "g"))

        sccs = digraph_strongly_connected_components(graph)
        self.assertListEqual([['a', 'c', 'b'], ['d'], ['f', 'g', 'h'], ['i', 'j']], sccs)

    @given(sets(elements=integers()))
    def test_topological_sort(self, nodes):
        nodes = list(nodes)
        graph = AdjListGraph(vertices=nodes)
        for i in range(1, 5):
            for u, v in zip(nodes, nodes[i:]):
                graph.add_directed(WeightedEdge(u, v))
        self.assertListEqual(nodes, list(topological_sort(graph)))
        self.assertListEqual(nodes, list(topological_sort_2(graph)))

    @given(sets(elements=integers(), min_size=2))
    def test_topological_sort_find_cycles(self, nodes):
        nodes = list(nodes)
        graph = AdjListGraph(vertices=nodes)
        for i in range(1, 5):
            for u, v in zip(nodes, nodes[i:]):
                graph.add_directed(WeightedEdge(u, v))

        indices = list(range(len(nodes)))
        u = np.random.choice(indices)
        v = np.random.choice(indices)
        u, v = max(u, v), min(u, v)
        graph.add_directed(WeightedEdge(nodes[u], nodes[v]))
        for algo in [topological_sort, topological_sort_2]:
            try:
                algo(graph)
                self.assertFalse("Excepted an cycle detected " + algo.__name__ + " " + str(graph))
            except CycleDetected:
                pass

    def test_articulation_points(self):
        nodes = [1, 2, 3, 4, 5, 6, 7]
        graph = AdjListGraph(vertices=nodes,
                             edges=[(1, 2), (2, 3), (1, 3),
                                    (5, 6), (6, 7), (5, 7),
                                    (3, 4), (4, 5)])
        # show_weighted_graph(graph)
        result = articulation_points(graph)
        self.assertSetEqual({3, 4, 5}, result, result)

    @given(sets(elements=integers()))
    @example(values={1, 2})
    def test_disjoint_set_becomes_joined(self, values):
        values = list(values)
        disjoint_set = DisjointSets(values)

        np.random.shuffle(values)
        for u, v in zip(values, values[1:]):
            self.assertFalse(disjoint_set.joined(u, v))

        for u, v in zip(values, values[1:]):
            disjoint_set.union(u, v)

        np.random.shuffle(values)
        for u, v in zip(values, values[1:]):
            self.assertTrue(disjoint_set.joined(u, v), repr(disjoint_set))

    def test_index_heap(self):
        heap = IndexHeap()
        for c in "ghaibcjdef":
            heap.add(c, ord(c))
        self.assertEqual('a', heap.min())
        heap.update('c', ord('a') - 1)
        self.assertEqual('c', heap.min())
        heap.pop_min()
        self.assertEqual('a', heap.min())

    @given(sets(elements=integers(), min_size=1))
    def test_heap_always_returns_minimum(self, values):
        heap = IndexHeap()

        for val in values:
            heap.add(key=val, priority=val*2)
        self.assertEqual(min(values), heap.min(), str(values) + " " + str(heap))

        for val in values:
            heap.update(key=val, priority=-1*val)
        self.assertEqual(max(values), heap.min(), str(values) + " " + str(heap))


    def minimum_spanning_tree(self, algorithm):
        """
        Example graph of
        https://en.wikipedia.org/wiki/Kruskal%27s_algorithm
        """
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

        result = algorithm(graph)
        weight = sum(e.weight for e in result)

        example_expected = {
            WeightedEdge(source='a', destination='d', weight=5),
            WeightedEdge(source='c', destination='e', weight=5),
            WeightedEdge(source='d', destination='f', weight=6),
            WeightedEdge(source='a', destination='b', weight=7),
            WeightedEdge(source='b', destination='e', weight=7),
            WeightedEdge(source='e', destination='g', weight=9)
        }
        expected_weight = sum(e.weight for e in example_expected)

        self.assertEqual(expected_weight, weight)
        # show_weighted_graph(graph)

    def test_kruskal(self):
        self.minimum_spanning_tree(kruskal)

    def test_prims_slow(self):
        self.minimum_spanning_tree(prims_slow)

    def test_prims(self):
        self.minimum_spanning_tree(prims)

    @given(sets(elements=integers()))
    def test_kruskal_versus_prims(self, nodes):
        nodes = list(nodes)

        # TODO - make a generator of graph instead (cause it hinders reproducibility)
        graph = AdjListGraph()
        for i in range(5):
            np.random.shuffle(nodes)
            for u, v in zip(nodes, nodes[1:]):
                graph.add(WeightedEdge(u, v, weight=np.random.randint(1, 10)))

        kruskal_result = sum(e.weight for e in kruskal(graph))
        prims_result = sum(e.weight for e in prims(graph))
        prims_slow_result = sum(e.weight for e in prims_slow(graph))
        self.assertEqual(kruskal_result, prims_result)
        self.assertEqual(kruskal_result, prims_slow_result)

    def test_dijkstra(self):
        # Example graph: https://www.baeldung.com/java-dijkstra
        vertices = list('abcdef')
        edges = [
            WeightedEdge('a', 'b', 10),
            WeightedEdge('a', 'c', 15),
            WeightedEdge('b', 'd', 12),
            WeightedEdge('b', 'f', 15),
            WeightedEdge('c', 'e', 10),
            WeightedEdge('d', 'f', 1),
            WeightedEdge('d', 'e', 2),
            WeightedEdge('f', 'e', 5)
        ]
        graph = AdjListGraph(vertices=vertices, edges=edges)
        distance = dijkstra(graph, 'a').shortest_distance_to('e')
        self.assertEqual(24, distance)
