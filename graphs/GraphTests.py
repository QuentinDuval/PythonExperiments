from graphs.DisjointSets import *
from graphs.Graphs import *
from graphs.Debugging import *

import numpy as np
import unittest

from hypothesis import given, example
from hypothesis.strategies import text
from hypothesis.strategies import lists, sets, integers


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

    def test_kruskal(self):
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
        result = set(kruskal(graph))
        expected = {
            WeightedEdge(source='a', destination='d', weight=5),
            WeightedEdge(source='c', destination='e', weight=5),
            WeightedEdge(source='d', destination='f', weight=6),
            WeightedEdge(source='a', destination='b', weight=7),
            WeightedEdge(source='b', destination='e', weight=7),
            WeightedEdge(source='e', destination='g', weight=9)
        }
        self.assertSetEqual(expected, result)
        # show_weighted_graph(graph)

    def test_index_heap(self):
        heap = IndexHeap()
        for c in "ghaibcjdef":
            heap.add(c, ord(c))
        self.assertEqual('a', heap.min())
        heap.update('c', ord('a') - 1)
        self.assertEqual('c', heap.min())
        heap.pop_min()
        self.assertEqual('a', heap.min())
