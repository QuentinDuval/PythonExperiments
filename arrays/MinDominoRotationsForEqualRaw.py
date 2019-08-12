"""
https://leetcode.com/problems/minimum-domino-rotations-for-equal-row/

In a row of dominoes, A[i] and B[i] represent the top and bottom halves of the i-th domino.
(A domino is a tile with two numbers from 1 to 6 - one on each half of the tile.)

We may rotate the i-th domino, so that A[i] and B[i] swap values.

Return the minimum number of rotations so that all the values in A are the same, or all the values in B are the same.

If it cannot be done, return -1.
"""

from typing import List


class Solution:
    def minDominoRotations_1(self, row1: List[int], row2: List[int]) -> int:
        """
        Idea:
        - if there is any combination possible, it is because one number is present in all pairs (row1[i], row2[i])
        - we can find these numbers (at most 2)
        - then we see on which side they occur more (and switch the other side)
        => we get the number
        1448 ms, beats 20 %
        """

        n = len(row1)
        if not row1 or not row2:
            return 0

        possibles = {}
        possibles[row1[0]] = [0, 0]
        possibles[row2[0]] = [0, 0]

        for i in range(n):
            self.intersect(possibles, row1[i], row2[i])
            if len(possibles) == 0:
                return -1

        min_moves = float('inf')
        for key, counts in possibles.items():
            up, lo = counts
            min_moves = min(min_moves, n - up, n - lo)
        return min_moves

    def intersect(self, possibles, hi, lo):
        for key in list(possibles.keys()):
            if key != lo and key != hi:
                del possibles[key]
            else:
                if key == hi:
                    possibles[key][0] += 1
                if key == lo:
                    possibles[key][1] += 1

    def minDominoRotations(self, row1: List[int], row2: List[int]) -> int:
        """
        Optimization: do two passes instead of one (one for each starting value)
        1280 ms, beats 76%
        """

        n = len(row1)
        if not row1 or not row2:
            return 0

        def check_for(val):
            up = 0
            lo = 0
            for i in range(n):
                if row1[i] != val and row2[i] != val:
                    return float('inf')
                if row1[i] == val:
                    up += 1
                if row2[i] == val:
                    lo += 1
            return min(n - up, n - lo)

        min_moves = float('inf')
        for val in row1[0], row2[0]:
            min_moves = min(min_moves, check_for(val))
        return min_moves if min_moves != float('inf') else -1
