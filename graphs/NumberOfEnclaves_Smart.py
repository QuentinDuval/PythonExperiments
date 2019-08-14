"""
https://leetcode.com/problems/number-of-enclaves

Given a 2D array A, each cell is 0 (representing sea) or 1 (representing land)

A move consists of walking from one land square 4-directionally to another land square, or off the boundary of the grid.

Return the number of land squares in the grid for which we cannot walk off the boundary of the grid in any number of moves.
"""
from typing import List


class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        """
        Do a dfs from the border to put to 0 all elements that are touched by the borders
        The simply count the number of 1s
        """

        if not grid or not grid[0]:
            return 0

        h = len(grid)
        w = len(grid[0])

        def dfs(i: int, j: int) -> bool:
            if grid[i][j] == 0:
                return

            to_visit = [(i, j)]
            grid[i][j] = 0
            while to_visit:
                i, j = to_visit.pop()
                for x, y in [(i + 1, j), (i - 1, j), (i, j - 1), (i, j + 1)]:
                    if 0 <= x < h and 0 <= y < w:
                        if grid[x][y] > 0:
                            grid[x][y] = 0
                            to_visit.append((x, y))

        for i in range(h):
            dfs(i, 0)
            dfs(i, w - 1)
        for j in range(w):
            dfs(0, j)
            dfs(h - 1, j)
        return sum(sum(row) for row in grid)
