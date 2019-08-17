"""
https://leetcode.com/problems/maximal-square/

Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.
"""


from typing import List


class Solution:
    def maximalSquare(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0

        h = len(grid)
        w = len(grid[0])

        tops = [[0] * w for _ in range(h)]
        lefts = [[0] * w for _ in range(h)]
        memo = [[0] * w for _ in range(h)]

        for i in range(h):
            filled = 1 if grid[i][0] == '1' else 0
            lefts[i][0] = filled
            memo[i][0] = filled

        for j in range(w):
            filled = 1 if grid[0][j] == '1' else 0
            tops[0][j] = filled
            memo[0][j] = filled

        for i in range(1, h):
            for j in range(1, w):
                if grid[i][j] == '1':
                    tops[i][j] = 1 + tops[i - 1][j]
                    lefts[i][j] = 1 + lefts[i][j - 1]
                    memo[i][j] = 1 + min(tops[i][j] - 1, lefts[i][j] - 1, memo[i - 1][j - 1])

        return max(cell for row in memo for cell in row) ** 2

# Better, and smarter:

class Solution:
    def maximalSquare(self, matrix):
        if not matrix or not matrix[0]:
            return 0

        h = len(matrix)
        w = len(matrix[0])

        memo = [[0] * w for _ in range(h)]

        for i in range(h):
            memo[i][0] = int(matrix[i][0])

        for j in range(w):
            memo[0][j] = int(matrix[0][j])

        for i in range(1, h):
            for j in range(1, w):
                if matrix[i][j] == "1":
                    memo[i][j] = 1 + min(memo[i - 1][j], memo[i][j - 1], memo[i - 1][j - 1])

        return max(memo[i][j] for j in range(w) for i in range(h)) ** 2