"""
https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/

Given a m * n grid, where each cell is either 0 (empty) or 1 (obstacle).
In one step, you can move up, down, left or right from and to an empty cell.

Return the minimum number of steps to walk from the upper left corner (0, 0)
to the lower right corner (m-1, n-1) given that you can eliminate at most k obstacles.

If it is not possible to find such walk return -1.
"""

from collections import *
from typing import List


class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        h = len(grid)
        w = len(grid[0])

        start_pos = (0, 0)
        discovered = {(start_pos, k)}
        to_visit = deque([(start_pos, k, 0)])

        def get_neighbors(i, j):
            if i < h - 1:
                yield i + 1, j
            if i > 0:
                yield i - 1, j
            if j < w - 1:
                yield i, j + 1
            if j > 0:
                yield i, j - 1

        while to_visit:
            pos, joker, dist = to_visit.popleft()
            if pos == (h - 1, w - 1):
                return dist

            for i, j in get_neighbors(*pos):
                if grid[i][j] == 0:
                    if ((i, j), joker) not in discovered:
                        discovered.add(((i, j), joker))
                        to_visit.append(((i, j), joker, 1 + dist))
                else:
                    if joker > 0 and ((i, j), joker - 1) not in discovered:
                        discovered.add(((i, j), joker - 1))
                        to_visit.append(((i, j), joker - 1, 1 + dist))

        return -1
