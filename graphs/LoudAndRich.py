"""
https://leetcode.com/problems/loud-and-rich/

In a group of N people (labelled 0, 1, 2, ..., N-1), each person has different amounts of money, and different levels of quietness.

For convenience, we'll call the person with label x, simply "person x".

We'll say that richer[i] = [x, y] if person x definitely has more money than person y.
Note that richer may only be a subset of valid observations.

Also, we'll say quiet[x] = q if person x has quietness q.

Now, return answer, where answer[x] = y if y is the least quiet person
(that is, the person y with the smallest value of quiet[y]),
among all people who definitely have equal to or more money than person x.
"""


from collections import defaultdict
from typing import List


class Solution:
    def loudAndRich(self, richer: List[List[int]], quiet: List[int]) -> List[int]:
        """
        The idea is to build a graph of people where edges represent "having more money" and
        then to traverse the graph starting from person 'x', to collect the "least quiet" one

        The edges will point in the direction "is richer than me" to help the traversal

        The problem is that we need to do this for all persons:
        - trying for every person would be awefully inefficient
        - instead, do a post order traversal (and get the minimum)
        - we need to cache the result for each person (it is not a tree)
        """

        graph = defaultdict(list)
        for rich, poor in richer:
            graph[poor].append(rich)

        loud_and_rich = {}

        def dfs(x: int) -> int:
            if x in loud_and_rich:
                return loud_and_rich[x]

            loud = x
            for neighbor in graph[x]:
                y = dfs(neighbor)
                if quiet[y] < quiet[loud]:
                    loud = y
            loud_and_rich[x] = loud
            return loud

        return [dfs(x) for x in range(len(quiet))]
