"""
https://leetcode.com/problems/minimum-falling-path-sum/

Given a square array of integers A, we want the minimum sum of a falling path through A.

A falling path starts at any element in the first row, and chooses one element from each row.
The next row's choice must be in a column that is different from the previous row's column by at most one.
"""


from functools import lru_cache
from typing import List


class Solution:
    def minFallingPathSum_top_down(self, grid: List[List[int]]) -> int:
        """
        Classing DP:
        - express row in terms of row - 1
        - add memoization O(H * W) sub-problems

        Complexity is O(H * W) = O(N)
        Beats 12% (172 ms)
        """
        h = len(grid)
        w = len(grid[0])

        @lru_cache(maxsize=None)
        def falling_path(row: int, col: int) -> int:
            if row == h:
                return 0
            if col < 0 or col >= w:
                return float('inf')

            sub_sol = min(falling_path(row + 1, col + i) for i in [-1, 0, 1])
            return grid[row][col] + sub_sol

        return min(falling_path(0, col) for col in range(w))

    def minFallingPathSum(self, grid: List[List[int]]) -> int:
        """
        Doing it bottom-up (+ realizing that it is same from top or bottom)
        Complexity in space falls to O(W)
        Beats 80% (132 ms)
        """
        h = len(grid)
        w = len(grid[0])

        memo = [0] * w
        for row in range(h):
            new_memo = [0] * w
            for col in range(w):
                sub_sol = memo[col]
                if col > 0:
                    sub_sol = min(sub_sol, memo[col - 1])
                if col < w - 1:
                    sub_sol = min(sub_sol, memo[col + 1])
                new_memo[col] = grid[row][col] + sub_sol
            memo = new_memo

        return min(memo)
