"""
https://leetcode.com/problems/reachable-nodes-in-subdivided-graph

Starting with an undirected graph (the "original graph") with nodes from 0 to N-1, subdivisions are made to some of the edges.

The graph is given as follows: edges[k] is a list of integer pairs (i, j, n) such that (i, j) is an edge of the original graph,

and n is the total number of new nodes on that edge.

Then, the edge (i, j) is deleted from the original graph, n new nodes (x_1, x_2, ..., x_n) are added to the original graph,

and n+1 new edges (i, x_1), (x_1, x_2), (x_2, x_3), ..., (x_{n-1}, x_n), (x_n, j) are added to the original graph.

Now, you start at node 0 from the original graph, and in each move, you travel along one edge.

Return how many nodes you can reach in at most M moves.
"""

import heapq
from typing import *


class Solution:
    def reachableNodes(self, edges: List[List[int]], M: int, N: int) -> int:
        """
        Really look like a kind of Dijsktra:
        - The question can be rephrased "how many nodes are at distance <= M"
        - With the 'new node' thingy being seen as the weight of an edge

        This only gives us the number of node before deletion.
        !!! We want the number of intermediary nodes as well !!!

        Dijkstra can do a kind of pre-processing:
        - gives us the time at which each node is integrated into the shortest path
        - then we can flood from here and compute the number of intermediary nodes in between
        """

        # Transformation to adjacency list
        nodes = {}
        graph = [{} for _ in range(N)]
        for x, y, d in edges:
            graph[x][y] = d + 1
            graph[y][x] = d + 1
            nodes[min(x, y), max(x, y)] = d

        # Dijkstra algorithm to compute the distances
        distances = {}
        to_visit = [(0, 0)]
        while to_visit:
            dist, node = heapq.heappop(to_visit)
            if dist > M: break
            if node in distances: continue

            distances[node] = dist
            for neighbor, weight in graph[node].items():
                if neighbor not in distances:
                    heapq.heappush(to_visit, (dist + weight, neighbor))

        # Post-processing phase:
        # - each edge is associated a potential that is diminished on both sides
        # - each time we diminish, we count the nodes
        total = len(distances)
        for a, dist in distances.items():
            for b, weight in graph[a].items():
                key = min(a, b), max(a, b)
                remaining = nodes[key]
                reached = min(M - dist, remaining)
                nodes[key] -= reached
                total += reached
        return total
