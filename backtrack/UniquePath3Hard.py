"""
https://leetcode.com/problems/unique-paths-iii

On a 2-dimensional grid, there are 4 types of squares:

* 1 represents the starting square.  There is exactly one starting square.
* 2 represents the ending square.  There is exactly one ending square.
* 0 represents empty squares we can walk over.
* -1 represents obstacles that we cannot walk over.

Return the number of 4-directional walks from the starting square to the ending square, that walk over every
non-obstacle square exactly once.
"""


from typing import List


class Solution:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        """
        The dimension of the grid is limited to height * width <= 20.

        We need to try every path that is of length == 2 + count (empty square)
        - that starts with the starting square
        - and ends with the ending square

        This seems to lend itself to back-tracking plus a fair amount of pruning
        No pruning here: 44ms
        """
        if not grid or not grid[0]:
            return 0

        h = len(grid)
        w = len(grid[0])

        empty_count = 0
        start = None

        for i in range(h):
            for j in range(w):
                if grid[i][j] == 0:
                    empty_count += 1
                elif grid[i][j] == 1:
                    start = i, j

        def neighbors(i, j):
            if i > 0:
                yield i - 1, j
            if i < h - 1:
                yield i + 1, j
            if j > 0:
                yield i, j - 1
            if j < w - 1:
                yield i, j + 1

        def backtrack(i, j, remaining):
            if remaining == 0:
                for x, y in neighbors(i, j):
                    if grid[x][y] == 2:
                        return 1
                return 0

            count = 0
            for x, y in neighbors(i, j):
                if grid[x][y] == 0:
                    grid[x][y] = -1
                    count += backtrack(x, y, remaining - 1)
                    grid[x][y] = 0
            return count

        return backtrack(start[0], start[1], empty_count)

    def uniquePathsIII_2(self, grid: List[List[int]]) -> int:
        """
        Note that pruning does not help here...
        It takes more time: 56ms.
        """
        if not grid or not grid[0]:
            return 0

        h = len(grid)
        w = len(grid[0])

        empty_count = 0
        start = None
        end = None

        for i in range(h):
            for j in range(w):
                if grid[i][j] == 0:
                    empty_count += 1
                elif grid[i][j] == 1:
                    start = i, j
                elif grid[i][j] == 2:
                    end = i, j

        def neighbors(i, j):
            if i > 0:
                yield i - 1, j
            if i < h - 1:
                yield i + 1, j
            if j > 0:
                yield i, j - 1
            if j < w - 1:
                yield i, j + 1

        def manhattan_distance_to_end(i, j):
            return abs(end[0] - i) + abs(end[1] - j)

        def backtrack(i, j, remaining):
            if manhattan_distance_to_end(i, j) > remaining + 1:
                return 0

            if remaining == 0:
                for x, y in neighbors(i, j):
                    if grid[x][y] == 2:
                        return 1
                return 0

            count = 0
            for x, y in neighbors(i, j):
                if grid[x][y] == 0:
                    grid[x][y] = -1
                    count += backtrack(x, y, remaining - 1)
                    grid[x][y] = 0
            return count

        return backtrack(start[0], start[1], empty_count)
