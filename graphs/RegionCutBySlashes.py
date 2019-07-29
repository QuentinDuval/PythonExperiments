"""
https://leetcode.com/problems/regions-cut-by-slashes

In a N x N grid composed of 1 x 1 squares, each 1 x 1 square consists of a /, \, or blank space.
These characters divide the square into contiguous regions.

(Note that backslash characters are escaped, so a \ is represented as "\\".)

Return the number of regions.
"""


from typing import List


class Solution:
    def regionsBySlashes(self, grid: List[str]) -> int:
        """
        Examples (number of cuts is not enough):

        " /"
        "  "
        => 1

        "/ "
        "  "
        => 2

        " /"
        " \"
        => 2

        " /"
        "/\"
        => 3

        " /"
        "//"
        => 3

        Connection with connected components?
        For a N * N grid, we need a 3N * 3N graph:
        - '/' cuts a link (disable cells on the diagnoal)
        - '\' cuts a link (disable cells on the diagonal)

        Other solution is to use UNION-FIND
        """

        h = len(grid)
        w = h

        matrix = [[True] * 3 * w for _ in range(3 * h)]
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                x = 3 * i
                y = 3 * j
                if cell == "/":
                    matrix[x + 2][y] = False
                    matrix[x + 1][y + 1] = False
                    matrix[x][y + 2] = False
                elif cell == "\\":
                    matrix[x][y] = False
                    matrix[x + 1][y + 1] = False
                    matrix[x + 2][y + 2] = False

        h = len(matrix)
        w = h

        def neighbors(x, y):
            if x > 0:
                yield x - 1, y
            if y > 0:
                yield x, y - 1
            if x < h - 1:
                yield x + 1, y
            if y < w - 1:
                yield x, y + 1

        def visit_from(x, y):
            to_visit = [(x, y)]
            while to_visit:
                x, y = to_visit.pop()
                matrix[x][y] = False  # Visited
                for i, j in neighbors(x, y):
                    if matrix[i][j]:
                        to_visit.append((i, j))

        count = 0
        for i in range(h):
            for j in range(w):
                if matrix[i][j]:
                    count += 1
                    visit_from(i, j)
        return count
