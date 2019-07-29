"""
https://leetcode.com/problems/couples-holding-hands/

N couples sit in 2N seats arranged in a row and want to hold hands. We want to know the minimum number of swaps so that every couple is sitting side by side. A swap consists of choosing any two people, then they stand up and switch seats.

The people and seats are represented by an integer from 0 to 2N-1, the couples are numbered in order, the first couple being (0, 1), the second couple being (2, 3), and so on with the last couple being (2N-2, 2N-1).

The couples' initial seating is given by row[i] being the value of the person who is initially sitting in the i-th seat.
"""


from typing import List


class Solution:
    def minSwapsCouples(self, row: List[int]) -> int:
        """
        Interesting swaps: a swap that increases the number of couples that are together
        => Hypothesis: we only need to do "interesting swaps" to find the minimum

        Several possiblities are equivalent (if you have to exchange between couple A and B,
        then either member of A will do fine - lead to same result).

        Example:

        [3, 1, 0, 5, 2, 4]
        = (1 and 2) =>
        [3, 2, 0, 5, 1, 4]
        = (5 and 1 - or 4 and 0) =>
        [3, 2, 0, 1, 5, 4]

        [3, 1, 0, 5, 2, 4]
        = (0 and 3) =>
        [0, 1, 3, 5, 2, 4]
        = (3 and 4 - or 2 and 5) =>
        [0, 1, 4, 5, 2, 3]

        Just pick a couple, and interchange with another couple, so that the number of
        reunited couples increases. Picking a direction will make it easy:
        - pick the first element that has no right partener beside
        - then interchange with the other couple half
        => O(N**2) time complexity (beat 90%)
        """

        def are_partners(i, j):
            return i // 2 == j // 2

        swaps = 0
        for i in range(0, len(row), 2):
            j = i + 1
            while j < len(row):
                if are_partners(row[i], row[j]):
                    break
                else:
                    j += 1

            if j != i + 1:
                row[i + 1], row[j] = row[j], row[i + 1]
                swaps += 1
        return swaps
