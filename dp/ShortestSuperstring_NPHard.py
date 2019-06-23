"""
https://leetcode.com/problems/find-the-shortest-superstring/

Given an array A of strings, find any smallest string that contains each string in A as a substring.

We may assume that no string in A is substring of another string in A.
"""


from functools import lru_cache
from typing import *


def shortestSuperstring(strings: List[str]) -> str:
    """
    Dynamic programming

    Reduce the problem of find the best concatenation for [1, 2, 3] to:
    - Choose 1 and recurse on [2, 3], and put 1 at the start or at the end of this sub-solution
    - Choose 2 and recurse on [1, 3], and put 2 at the start or at the end of this sub-solution
    - Etc
    """

    def concat(s1, s2):
        l = min(len(s1), len(s2))
        for i in reversed(range(1, l + 1)):
            if s1.endswith(s2[:i]):
                return s1 + s2[i:]
        return s1 + s2

    @lru_cache(maxsize=None)
    def visit(visited):
        best_score = float('inf')
        best_suffix = ""

        for i, s in enumerate(strings):
            if not (1 << i) & visited:
                sub_suffix = visit((1 << i) | visited)

                suffix = concat(s, sub_suffix)
                if len(suffix) < best_score:
                    best_score = len(suffix)
                    best_suffix = suffix

                suffix = concat(sub_suffix, s)
                if len(suffix) < best_score:
                    best_score = len(suffix)
                    best_suffix = suffix

        return best_suffix

    return visit(0)
