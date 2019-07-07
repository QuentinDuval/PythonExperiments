"""
https://leetcode.com/problems/pyramid-transition-matrix/

We are stacking blocks to form a pyramid. Each block has a color which is a one letter string.

We are allowed to place any color block C on top of two adjacent blocks of colors A and B, if and only if ABC is an allowed triple.

We start with a bottom row of bottom, represented as a single string. We also start with a list of allowed triples allowed.
Each allowed triple is represented as a string of length 3.

Return true if we can build the pyramid all the way to the top, otherwise false.
"""


from collections import defaultdict
from typing import List


class Solution:
    def pyramidTransition(self, bottom: str, allowed: List[str]) -> bool:
        """
        Try systematically all solutions (back-tracking)
        Pre-process the allowed to allow fast search

        Beats 96%
        """

        graph = defaultdict(list)
        for x in allowed:
            graph[x[:2]].append(x[2])

        def all_possible(stage: str, partial: str, pos: int):
            if pos == len(stage) - 1:
                yield partial
                return

            for x in graph[stage[pos:pos + 2]]:
                yield from all_possible(stage, partial + x, pos + 1)

        def backtrack(stage: str) -> bool:
            if len(stage) == 1:
                return True

            for possible in all_possible(stage, "", 0):
                if backtrack(possible):
                    return True
            return False

        return backtrack(bottom)
