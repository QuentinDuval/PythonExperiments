"""
https://leetcode.com/problems/range-sum-query-2d-immutable/

Given a 2D matrix matrix, find the sum of the elements inside the rectangle defined
by its upper left corner (row1, col1) and lower right corner (row2, col2).
"""

from typing import List


class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        self.cum_sum = [[] for row in matrix]
        for i, row in enumerate(matrix):
            row_cum_sum = 0
            for j, val in enumerate(row):
                row_cum_sum += val
                prev_row = self.cum_sum[i-1][j] if i > 0 else 0
                self.cum_sum[i].append(prev_row + row_cum_sum)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        total = self.cum_sum[row2][col2]
        top_left = self.cum_sum[row1-1][col1-1] if row1 * col1 > 0 else 0
        top = self.cum_sum[row1-1][col2] if row1 > 0 else 0
        left = self.cum_sum[row2][col1-1] if col1 > 0 else 0
        return total + top_left - top - left
