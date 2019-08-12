"""
https://leetcode.com/problems/possible-bipartition

Given a set of N people (numbered 1, 2, ..., N), we would like to split everyone into two groups of any size.

Each person may dislike some other people, and they should not go into the same group.

Formally, if dislikes[i] = [a, b], it means it is not allowed to put the people numbered a and b into the same group.

Return true if and only if it is possible to split everyone into two groups in this way.
"""

from collections import defaultdict
from typing import List


class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        """
        This is a classic graph problem:
        - We try to see if we can create 2 groups of nodes such that there are
          no (dislike) connections between any people of the same group.
        - This is different from the connectivity problem (in which we unify by edge).

        Idea:
        - Create an adjacency list
        - Visit the graph and tag each node 0 or 1 alternatively
        - Check neighbors: if they have the same tag, this is bad

        Beats 98%
        """

        graph = defaultdict(list)
        for a, b in dislikes:
            graph[a].append(b)
            graph[b].append(a)

        tags = {}
        for start in range(1, n + 1):
            if start in tags:
                continue

            tags[start] = 0
            to_visit = [start]
            while to_visit:
                node = to_visit.pop()
                tag = tags[node]
                for neighbor in graph[node]:
                    if neighbor not in tags:
                        tags[neighbor] = 1 - tag  # Alternates between 0 and 1
                        to_visit.append(neighbor)
                    elif tags[neighbor] != 1 - tag:
                        return False
        return True
