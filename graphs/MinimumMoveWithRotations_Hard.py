"""
https://leetcode.com/problems/minimum-moves-to-reach-target-with-rotations

In an n*n grid, there is a snake that spans 2 cells and starts moving from the top left corner at (0, 0) and (0, 1). The grid has empty cells represented by zeros and blocked cells represented by ones. The snake wants to reach the lower right corner at (n-1, n-2) and (n-1, n-1).

In one move the snake can:

* Move one cell to the right if there are no blocked cells there. This move keeps the horizontal/vertical position of the snake as it is.
* Move down one cell if there are no blocked cells there. This move keeps the horizontal/vertical position of the snake as it is.
* Rotate clockwise if it's in a horizontal position and the two cells under it are both empty. In that case the snake moves from (r, c) and (r, c+1) to (r, c) and (r+1, c).
* Rotate counterclockwise if it's in a vertical position and the two cells to its right are both empty. In that case the snake moves from (r, c) and (r+1, c) to (r, c) and (r, c+1).

Return the minimum number of moves to reach the target.

If there is no way to reach the target, return -1.
"""

from collections import deque
from typing import List


class Solution:
    def minimumMoves(self, grid: List[List[int]]) -> int:
        """
        Not that hard, a BFS does the trick.
        The only trick is to consider the state as tail and head position of the snake: ((tx,ty),(hx,hy)) below

        Beat 44%.
        """
        if not grid or not grid[0]:
            return -1

        h = len(grid)
        w = len(grid[0])
        start = ((0, 0), (0, 1))
        goal = ((h - 1, w - 2), (h - 1, w - 1))

        def neighbors(tx, ty, hx, hy):

            # Horizontal position
            if tx == hx:
                if tx < h - 1 and grid[tx + 1][ty] == 0 and grid[hx + 1][hy] == 0:
                    yield (tx, ty), (tx + 1, ty)  # Rotate down
                if hy < w - 1 and grid[hx][hy + 1] == 0:
                    yield (hx, hy), (hx, hy + 1)  # Move left
                if tx < h - 1 and grid[tx + 1][ty] == 0 and grid[hx + 1][hy] == 0:  # Move down
                    yield (tx + 1, ty), (hx + 1, hy)

            # Vertical position
            else:
                if ty < w - 1 and grid[tx][ty + 1] == 0 and grid[hx][hy + 1] == 0:
                    yield (tx, ty), (tx, ty + 1)  # Rotate up
                if hx < h - 1 and grid[hx + 1][hy] == 0:
                    yield (hx, hy), (hx + 1, hy)  # Move down
                if ty < w - 1 and grid[tx][ty + 1] == 0 and grid[hx][hy + 1] == 0:  # Move left
                    yield (tx, ty + 1), (hx, hy + 1)

        discovered = {start}
        queue = deque([(start, 0)])
        while queue:
            node, d = queue.popleft()
            if node == goal:
                return d

            (tx, ty), (hx, hy) = node
            for neigh in neighbors(tx, ty, hx, hy):
                if neigh not in discovered:
                    discovered.add(neigh)
                    queue.append((neigh, d + 1))

        return -1
