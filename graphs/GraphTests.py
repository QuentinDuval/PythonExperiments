from graphs.Graphs import *

import numpy as np
import unittest


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

