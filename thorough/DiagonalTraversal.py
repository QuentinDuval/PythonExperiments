"""
https://leetcode.com/problems/diagonal-traverse/
"""


from typing import List


class Solution:
    def findDiagonalOrder(self, matrix: List[List[int]]) -> List[int]:
        """
        Idea is to use the L1 distance, and iterate, increasing the L1 distance
        But the hard part is to get the indexes right (and not do a 'if in matrix' to avoid wasting CPU cycles)
        """

        if not matrix or not matrix[0]:
            return []

        h = len(matrix)
        w = len(matrix[0])
        traversal = []

        for l1_distance in range(h + w):
            min_j = max(0, l1_distance - (h - 1))
            max_j = min(w, l1_distance + 1)
            rng = range(min_j, max_j)
            if l1_distance % 2:
                rng = reversed(rng)
            traversal.extend(matrix[l1_distance - j][j] for j in rng)

        return traversal
