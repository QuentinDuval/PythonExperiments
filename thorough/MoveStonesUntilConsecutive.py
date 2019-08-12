"""
https://leetcode.com/problems/moving-stones-until-consecutive/

Three stones are on a number line at positions a, b, and c.

Each turn, you pick up a stone at an endpoint (ie., either the lowest or highest position stone), and move it to an unoccupied position between those endpoints.  Formally, let's say the stones are currently at positions x, y, z with x < y < z.  You pick up the stone at either position x or position z, and move that stone to an integer position k, with x < k < z and k != y.

The game ends when you cannot make any more moves, ie. the stones are in consecutive positions.

When the game ends, what is the minimum and maximum number of moves that you could have made?  Return the answer as an length 2 array: answer = [minimum_moves, maximum_moves]
"""
from typing import List


class Solution:
    def numMovesStones(self, a: int, b: int, c: int) -> List[int]:
        """
        The key is to consider the fact that if a < b < c, then only a and c can be moved.
        """

        stones = list(sorted([a, b, c]))
        print(stones)
        if stones[0] + 2 == stones[2]:
            return [0, 0]
        return [self.min_moves(*stones), self.max_moves(*stones)]

    def min_moves(self, a, b, c):
        """
        It is always solvable in 1 or 2 moves:
        - 1 move if the gap between the two closest stones is below 1
        - 2 moves otherwise
        """
        if a + 1 == b or b + 1 == c:
            return 1
        if a + 2 == b or b + 2 == c:
            return 1
        return 2

    def max_moves(self, a, b, c):
        """
        Move a toward b, slowly
        Move c toward b, slowly
        """
        return (b - a - 1) + (c - b - 1)
