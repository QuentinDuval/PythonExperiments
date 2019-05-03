from typing import List


"""
https://leetcode.com/problems/rotate-image/submissions/
"""


def rotate(matrix: List[List[int]]) -> None:
    n = len(matrix)
    if n <= 1:
        return

    def next_coordinate(depth, x, y):
        if x == depth and y < n - 1 - depth:
            return (x, y + 1)
        if y == n - 1 - depth and x < n - 1 - depth:
            return (x + 1, y)
        if x == n - 1 - depth and y > depth:
            return (x, y - 1)
        if y == depth and x > depth:
            return (x - 1, y)
        return None

    max_depth = (n + 1) // 2
    for depth in range(0, max_depth + 1):
        x1, y1 = (depth, depth)
        x2, y2 = (depth, n - 1 - depth)
        x3, y3 = (n - 1 - depth, n - 1 - depth)
        x4, y4 = (n - 1 - depth, depth)

        while y1 < n - 1 - depth:
            temp, matrix[x2][y2] = matrix[x2][y2], matrix[x1][y1]
            # print(matrix, temp)
            temp, matrix[x3][y3] = matrix[x3][y3], temp
            # print(matrix, temp)
            temp, matrix[x4][y4] = matrix[x4][y4], temp
            # print(matrix, temp)
            matrix[x1][y1] = temp
            # print(matrix)

            x1, y1 = next_coordinate(depth, x1, y1)
            x2, y2 = next_coordinate(depth, x2, y2)
            x3, y3 = next_coordinate(depth, x3, y3)
            x4, y4 = next_coordinate(depth, x4, y4)
