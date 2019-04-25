"""
https://leetcode.com/problems/out-of-boundary-paths/

There is an m by n grid with a ball. Given the start coordinate (i,j) of the ball, you can move the ball to adjacent cell or cross the grid boundary in four directions (up, down, left, right). However, you can at most move N times. Find out the number of paths to move the ball out of grid boundary. The answer may be very large, return it after mod 109 + 7.
"""

from functools import lru_cache


def trace(f):
    def traced(*args):
        out = f(*args)
        print(args, "=>", out)
        return out

    return traced


class Solution:
    def findPaths(self, h: int, w: int, N: int, i: int, j: int) -> int:
        """
        Nothing forbids to move the ball back where it went

        Dynamic programming seems possible.

        Number of sub-solutions: O(N^3)
        - N for x axis
        - N for y axis
        - N for remaining moves

        Top-down dynamic programming is better since we do not know which sub-solutions are needed
        """

        @lru_cache(maxsize=None)
        # @trace
        def visit(i, j, moves):
            if moves == 0:
                return 0

            if i < 0 or i >= h or j < 0 or j >= w:
                return 1

            if moves == 1:
                count = 0
                if i == 0:
                    count += 1
                if i == h - 1:
                    count += 1
                if j == 0:
                    count += 1
                if j == w - 1:
                    count += 1
                return count

            return visit(i - 1, j, moves - 1) + visit(i + 1, j, moves - 1) + visit(i, j - 1, moves - 1) + visit(i, j + 1, moves - 1)

        return visit(i, j, N) % (10 ** 9 + 7)
