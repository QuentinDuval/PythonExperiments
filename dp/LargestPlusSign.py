"""
https://leetcode.com/problems/largest-plus-sign

In a 2D grid from (0, 0) to (N-1, N-1), every cell contains a 1, except those cells in the given list mines which are 0.
What is the largest axis-aligned plus sign of 1s contained in the grid? Return the order of the plus sign.
If there is none, return 0.
"""
from typing import List


class Solution:
    def orderOfLargestPlusSign(self, n: int, mines: List[List[int]]) -> int:
        """
        Brutal, with DP, beats 70%
        """
        if n == 0:
            return 0

        h = w = n
        grid = [[1] * w for _ in range(h)]
        for x, y in mines:
            grid[x][y] = 0

        tops = [[0] * w for _ in range(h)]
        bots = [[0] * w for _ in range(h)]
        lefts = [[0] * w for _ in range(h)]
        rights = [[0] * w for _ in range(h)]

        for i in range(h):
            for j in range(w):
                if grid[i][j]:
                    tops[i][j] = 1 + (tops[i - 1][j] if i > 0 else 0)
                    lefts[i][j] = 1 + (lefts[i][j - 1] if j > 0 else 0)

        for i in reversed(range(h)):
            for j in reversed(range(w)):
                if grid[i][j]:
                    bots[i][j] = 1 + (bots[i + 1][j] if i < h - 1 else 0)
                    rights[i][j] = 1 + (rights[i][j + 1] if j < w - 1 else 0)

        def plus_at(i, j):
            return min(tops[i][j], bots[i][j], lefts[i][j], rights[i][j])

        return max(plus_at(i, j) for i in range(h) for j in range(w))



