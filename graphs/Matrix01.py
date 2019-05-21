"""
https://leetcode.com/problems/01-matrix/

Given a matrix consists of 0 and 1, find the distance of the nearest 0 for each cell.
The distance between two adjacent cells is 1.
"""

from collections import deque
from typing import List


class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        """
        Start with a queue of all the zeros, and expand the space around
        Using the trick of updating the input matrix to track what is visited
        """
        if not matrix or not matrix[0]:
            return []

        h = len(matrix)
        w = len(matrix[0])

        def neighbors(i, j):
            if i > 0:
                yield (i - 1, j)
            if i < h - 1:
                yield (i + 1, j)
            if j > 0:
                yield (i, j - 1)
            if j < w - 1:
                yield (i, j + 1)

        to_visit = deque()
        for i in range(h):
            for j in range(w):
                if matrix[i][j] == 0:
                    to_visit.append((i, j, 0))

        distances = [[0] * w for _ in range(h)]
        while to_visit:
            i, j, d = to_visit.popleft()
            distances[i][j] = d
            for ni, nj in neighbors(i, j):
                if matrix[ni][nj] != 0:
                    matrix[ni][nj] = 0
                    to_visit.append((ni, nj, d + 1))
        return distances
