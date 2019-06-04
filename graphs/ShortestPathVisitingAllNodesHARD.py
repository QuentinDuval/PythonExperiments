"""
https://leetcode.com/problems/shortest-path-visiting-all-nodes/

An undirected, connected graph of N nodes (labeled 0, 1, 2, ..., N-1) is given as graph.

graph.length = N, and j != i is in the list graph[i] exactly once, if and only if nodes i and j are connected.

Return the length of the shortest path that visits every node.

Note:
You may start and stop at any node, you may revisit nodes multiple times, and you may reuse edges.
"""


from typing import List
from collections import deque, defaultdict


class Solution:
    def shortestPathLength(self, g_list: List[List[int]]) -> int:
        """
        If naive BFS, then we can't visit the visited node again, can't solve this issue
        The key is to define the state of visited as (cur_node, visited_nodes)
        If next to explore is the same node and same visisted then it is a loop, we then won't visit in BFS algo

        BRILLIANT
        """

        graph = {}
        n = len(g_list)
        done = (1 << n) - 1
        for i, targets in enumerate(g_list):
            graph[i] = targets
        queue = deque()

        # node => visited set mapping
        visited = defaultdict(set)

        # Add all nodes to initial queue
        for i in range(n):
            # Use bit vector to represent visited nodes
            queue.appendleft((0, i, 1 << i))

        # BFS
        while queue:
            dist, cur_node, state = queue.pop()
            if state == done: return dist
            for next_node in graph[cur_node]:
                if state not in visited[next_node]:
                    visited[next_node].add(state)
                    queue.appendleft((dist + 1, next_node, state | 1 << next_node))
        return -1
