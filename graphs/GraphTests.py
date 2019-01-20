from graphs.DisjointSets import *
from graphs.Graphs import *

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
