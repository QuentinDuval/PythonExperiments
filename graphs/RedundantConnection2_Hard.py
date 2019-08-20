"""
https://leetcode.com/problems/redundant-connection-ii

In this problem, a rooted tree is a directed graph such that, there is exactly one node (the root)
for which all other nodes are descendants of this node, plus every node has exactly one parent,
except for the root node which has no parents.

The given input is a directed graph that started as a rooted tree with N nodes (with distinct values 1, 2, ..., N),
with one additional directed edge added. The added edge has two different vertices chosen from 1 to N,
and was not an edge that already existed.

The resulting graph is given as a 2D-array of edges. Each element of edges is a pair [u, v] that represents
a directed edge connecting nodes u and v, where u is a parent of child v.

Return an edge that can be removed so that the resulting graph is a rooted tree of N nodes.

If there are multiple answers, return the answer that occurs last in the given 2D-array.
"""

from collections import defaultdict
from typing import List


class Solution:
    def findRedundantDirectedConnection(self, edges: List[List[int]]) -> List[int]:
        """
        If there is a double parent for a node:
        - one of the edge from one of these parent is good
        - the other one might be good as well (or disconnect the graph)

        Example:

        a -> b <- e
             |    ^
             v    |
             c -> d

        'b' has two parents, but only (e,b) can be removed:
        - no other letter than 'a' can be the root
        - because we enter the cycle via 'a'

        There might be cases in which there are no double parent for each nodes.
        In such cases, we have a cycle:

        a <- d -> e
        |    ^
        v    |
        b -> c

        Which edge to remove in the cycle is not important, all of these are equivalent.
        - We can remove (a, b) => root will be 'b'
        - We can remove (d, a) => root will be 'a'
        - Etc

        There are also cases in which there are no cycle, just 2 parents.

        All of this is accounted below, beats 90%.
        """

        nodes = set()
        graph = defaultdict(list)  # To find the cycle
        igraph = defaultdict(list)  # To find incoming edges
        for u, v in edges:
            graph[u].append(v)
            igraph[v].append(u)
            nodes.add(u)
            nodes.add(v)

        for start_node in nodes:

            # If a node has two parents
            if len(igraph[start_node]) >= 2:

                # Either it comes from a cycle => eliminate the edge that is in the cycle
                cycle = self.find_cycle(graph, set(), start_node)
                if cycle:
                    for node in cycle:
                        if len(igraph[node]) >= 2:
                            for parent in igraph[node]:
                                if parent in cycle:
                                    return [parent, node]

                # Or it does not matter which edge to remove => choose the last one in the list
                else:
                    highest_index = -1
                    for parent in igraph[start_node]:
                        index = edges.index([parent, start_node])
                        highest_index = max(highest_index, index)
                    return edges[highest_index]

        # If there are no nodes with two parents
        discovered = set()
        for start_node in nodes:
            if start_node in discovered:
                continue

            # Find a cycle
            cycle = self.find_cycle(graph, discovered, start_node)
            if cycle:
                # Remove the last edge in the list that is in the cycle
                last_edge = None
                for u, v in edges:
                    if u in cycle and v in cycle:
                        last_edge = [u, v]
                return last_edge

        return None

    def find_cycle(self, graph, discovered, start_node):
        on_stack = set()
        to_visit = [('visit', start_node)]
        discovered.add(start_node)
        while to_visit:
            step, node = to_visit.pop()
            if step == 'pop':
                on_stack.remove(node)
            else:
                on_stack.add(node)
                to_visit.append(('pop', node))
                for neighbor in graph[node]:
                    if neighbor in on_stack:
                        return on_stack
                    if neighbor not in discovered:
                        to_visit.append(('visit', neighbor))
                        discovered.add(neighbor)
        return set()



