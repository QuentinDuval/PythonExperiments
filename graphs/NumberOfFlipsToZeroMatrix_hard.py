"""
https://leetcode.com/problems/minimum-number-of-flips-to-convert-binary-matrix-to-zero-matrix/

Given a m x n binary matrix mat. In one step, you can choose one cell and flip it and all the four neighbours of it if they exist (Flip is changing 1 to 0 and 0 to 1). A pair of cells are called neighboors if they share one edge.

Return the minimum number of steps required to convert mat to a zero matrix or -1 if you cannot.

Binary matrix is a matrix with all cells equal to 0 or 1 only.

Zero matrix is a matrix with all cells equal to 0.
"""


from collections import deque
from typing import List


class Solution:
    def minFlips(self, mat: List[List[int]]) -> int:
        h = len(mat)
        w = len(mat[0])

        def as_bits(mat: List[List[int]]) -> int:
            bits = 0
            for i in range(h):
                for j in range(w):
                    if mat[i][j] == 1:
                        bits |= 1 << (w * i + j)
            return bits

        def flip(node, i, j):
            pos = w * i + j
            node ^= 1 << pos
            if i < h - 1:
                node ^= 1 << (pos + w)
            if i > 0:
                node ^= 1 << (pos - w)
            if j < w - 1:
                node ^= 1 << (pos + 1)
            if j > 0:
                node ^= 1 << (pos - 1)
            return node

        def get_neighbors(node):
            for i in range(h):
                for j in range(w):
                    yield flip(node, i, j)

        start_node = as_bits(mat)
        discovered = {start_node}
        to_visit = deque([(start_node, 0)])
        while to_visit:
            node, dist = to_visit.popleft()
            if node == 0:
                return dist
            for neigh in get_neighbors(node):
                if neigh not in discovered:
                    discovered.add(neigh)
                    to_visit.append((neigh, 1 + dist))
        return -1
