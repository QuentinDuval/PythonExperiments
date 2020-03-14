"""
https://leetcode.com/problems/minimum-score-triangulation-of-polygon/
"""

from functools import lru_cache
from typing import List


class Solution:
    def minScoreTriangulation(self, vertices: List[int]) -> int:

        @lru_cache(maxsize=None)
        def visit(lo: int, hi: int) -> int:
            if lo + 1 >= hi:
                return 0

            min_cost = float('inf')
            for k in range(lo + 1, hi):
                triangle_cost = vertices[lo] * vertices[k] * vertices[hi]
                min_cost = min(min_cost, visit(lo, k) + visit(k, hi) + triangle_cost)
            return min_cost

        return visit(0, len(vertices) - 1)

