"""
https://leetcode.com/problems/matrix-block-sum

Given a m * n matrix mat and an integer K, return a matrix answer where each answer[i][j] is the sum of all elements
mat[r][c] for i - K <= r <= i + K, j - K <= c <= j + K, and (r, c) is a valid position in the matrix.
"""


from typing import List


class Solution:
    def matrixBlockSum(self, matrix: List[List[int]], k: int) -> List[List[int]]:
        if not matrix or not matrix[0]:
            return matrix

        # Rework the matrix to contains the sum of the upper-left matrix
        h = len(matrix)
        w = len(matrix[0])
        for i in range(h):
            prefix_sum = 0
            for j in range(w):
                top = matrix[i - 1][j] if i > 0 else 0
                prefix_sum += matrix[i][j]
                matrix[i][j] = prefix_sum + top

        # Access the matrix with out-of-bounds
        def at(i, j):
            if i < 0 or j < 0: return 0
            i = min(i, h - 1)
            j = min(j, w - 1)
            return matrix[i][j]

        # Use these prefix sums in order to return the block sums
        blocks = [[0] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                bottom_right = at(i + k, j + k)
                left = at(i + k, j - k - 1)
                top = at(i - k - 1, j + k)
                top_left = at(i - k - 1, j - k - 1)
                blocks[i][j] = bottom_right - left - top + top_left
        return blocks

