"""
https://leetcode.com/problems/search-a-2d-matrix-ii

Write an efficient algorithm that searches for a value in an m x n matrix.

This matrix has the following properties:
- Integers in each row are sorted in ascending from left to right.
- Integers in each column are sorted in ascending from top to bottom.
"""


from typing import List


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int):
        """
        Search at two positions by iterations:
        [
          [1,   4,  7, 11, 15],
          [2,   5,  8, 12, 19],
          [3,   6,  9, 16, 22],
          [10, 13, 14, 17, 24],
          [18, 21, 23, 26, 30]
                   ^
                   |
             search at 9 (compare against target)

        If target is lower: it cannot be in bottom-right.
        If target is higher: it cannot be in upper-left.
        => recurse in 3 problems of size / 4.


        Time complexity is O(N ^ log based 4 of 3)
        Beats 86%.

        !!! IMPORTANT !!!
        Time complexity is not log base 4/3 of N (it is not the same as recuring to one problem of size 3/4 N)

        Trace the recusion tree:
        - It has log base 4 of N depth
        - It has 3 ^ H width at the end
        - The cost of each node is O(1)

        So the cost is:
        3 ^ (log base 4 of N)
        => 3 ^ (log base 4 of 3 * log base 3 of N)
        => 3 ^ (log base 3 of N) ^ (log base 4 of 3)
        => N ^ (log base 4 of 3)

        Alternatively, you can use the Master Theorem
        """

        if not matrix or not matrix[0]:
            return False

        h = len(matrix)
        w = len(matrix[0])

        def search(min_i, max_i, min_j, max_j):
            if min_i > max_i or min_j > max_j:
                return False

            mid_i = min_i + (max_i - min_i) // 2
            mid_j = min_j + (max_j - min_j) // 2
            mid_val = matrix[mid_i][mid_j]
            if mid_val == target:
                return True

            # eliminate top-left corner
            if mid_val < target:
                found = search(mid_i + 1, max_i, mid_j + 1, max_j)  # bottom-right
                found |= search(mid_i + 1, max_i, min_j, mid_j)     # bottom-left
                found |= search(min_i, mid_i, mid_j + 1, max_j)     # top-right
                return found

            # eliminate bottom-right corder
            else:
                found = search(min_i, mid_i - 1, min_j, mid_j - 1)  # top-left
                found |= search(mid_i, max_i, min_j, mid_j - 1)     # bottom-left
                found |= search(min_i, mid_i - 1, mid_j, max_j)     # top-right
                return found

        return search(0, h - 1, 0, w - 1)
