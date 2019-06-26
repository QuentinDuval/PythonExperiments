"""
https://leetcode.com/problems/is-graph-bipartite

Given an undirected graph, return true if and only if it is bipartite.

Recall that a graph is bipartite if we can split it's set of nodes into two independent subsets A and B such that every edge in the graph has one node in A and another node in B.

The graph is given in the following form: graph[i] is a list of indexes j for which the edge between nodes i and j exists.  Each node is an integer between 0 and graph.length - 1.  There are no self edges or parallel edges: graph[i] does not contain i, and it doesn't contain any element twice.
"""


from typing import List


class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        """
        Do a Depth First Search:
        - Along the search, track the "time" the node has been discovered
        - Each time you find the node gain, check that the parity of the new "time" is the same
        """
        if not graph:
            return False

        discovery_time = {}
        for start in range(len(graph)):
            if start in discovery_time:
                continue

            to_visit = [(start, 0)]
            discovery_time[start] = 0

            while to_visit:
                node, time = to_visit.pop()
                for neighbor in graph[node]:
                    if neighbor not in discovery_time:
                        discovery_time[neighbor] = time + 1
                        to_visit.append((neighbor, time + 1))
                    elif discovery_time[neighbor] % 2 == time % 2:
                        return False
        return True

