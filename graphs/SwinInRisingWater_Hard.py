"""
https://leetcode.com/problems/swim-in-rising-water/

On an N x N grid, each square grid[i][j] represents the elevation at that point (i,j).

Now rain starts to fall. At time t, the depth of the water everywhere is t.
You can swim from a square to another 4-directionally adjacent square if and only if the elevation of BOTH squares
individually are at most t. You can swim infinite distance in zero time.

Of course, you must stay within the boundaries of the grid during your swim.

You start at the top left square (0, 0). What is the least time until you can reach the bottom right square (N-1, N-1)?
"""


import heapq
from typing import List


class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        """
        To be able to go from X to Y, the water must be at the very least max(X, Y).
        => The goal is to minimize the value of the cell we use to go the other side.

        This is kind of a Dijstra, but not really:
        - the edge are not interesting (we care about values of cell)
        - we do not miminize a sum but a max

        But we can think of it as Dijstra because the algorithm is GREEDY:
        - just explore the cell with the least value (use min-heap)
        - it might not be on the path, but it does not matter
        """

        n = len(grid)

        to_visit = []

        def add_to_visit(i, j):
            heapq.heappush(to_visit, (grid[i][j], i, j))

        def neighbors(i, j):
            if i > 0:
                yield i - 1, j
            if j > 0:
                yield i, j - 1
            if i < n - 1:
                yield i + 1, j
            if j < n - 1:
                yield i, j + 1

        min_depth = 0
        visited = set()
        add_to_visit(0, 0)
        while to_visit:
            depth, i, j = heapq.heappop(to_visit)
            if (i, j) in visited:
                continue

            min_depth = max(min_depth, depth)
            if (i, j) == (n - 1, n - 1):
                break

            visited.add((i, j))
            for x, y in neighbors(i, j):
                if (x, y) not in visited:
                    add_to_visit(x, y)
        return min_depth

