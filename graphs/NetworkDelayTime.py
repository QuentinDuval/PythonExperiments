"""
https://leetcode.com/problems/network-delay-time

There are N network nodes, labelled 1 to N.

Given times, a list of travel times as directed edges times[i] = (u, v, w), where u is the source node,
v is the target node, and w is the time it takes for a signal to travel from source to target.

Now, we send a signal from a certain node K. How long will it take for all nodes to receive the signal?
If it is impossible, return -1.
"""

from collections import defaultdict
from typing import List
import heapq


class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, start_node: int) -> int:
        """
        This is Dijsktra
        """

        graph = defaultdict(list)
        for u, v, delay in times:
            graph[u].append((v, delay))

        visited = {}
        to_visit = [(0, start_node)]
        while to_visit:
            total_delay, node = heapq.heappop(to_visit)
            if node in visited:
                continue

            visited[node] = total_delay
            for neighbor, delay in graph[node]:
                if neighbor not in visited:
                    heapq.heappush(to_visit, (total_delay + delay, neighbor))

        if len(visited) != n:
            return -1
        else:
            return max(visited.values(), default=0)
