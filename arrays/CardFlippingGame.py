"""
https://leetcode.com/problems/card-flipping-game/

On a table are N cards, with a positive integer printed on the front and back of each card (possibly different).

We flip any number of cards, and after we choose one card.

If the number X on the back of the chosen card is not on the front of any card, then this number X is good.

What is the smallest number that is good?  If no number is good, output 0.

Here, fronts[i] and backs[i] represent the number on the front and back of card i.

A flip swaps the front and back numbers, so the value on the front is now on the back and vice versa.
"""


from typing import List


class Solution:
    def flipgame(self, fronts: List[int], backs: List[int]) -> int:
        """
        If a number is both on front and back, it cannot be selected.
        If not, there is always a way to move all to back the smallest number.

        So you can just move through the fronts and back and keep track of the
        minimum number that is not both on a front and back.
        => O(N) complexity is possible in two passes
        """

        banned = set()
        n = len(fronts)
        for i in range(n):
            if fronts[i] == backs[i]:
                banned.add(fronts[i])

        min_card = float('inf')
        for i in range(n):
            if fronts[i] not in banned:
                min_card = min(min_card, fronts[i])
            if backs[i] not in banned:
                min_card = min(min_card, backs[i])
        return min_card if min_card != float('inf') else 0
