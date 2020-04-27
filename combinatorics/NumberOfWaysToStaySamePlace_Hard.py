"""
https://leetcode.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/

You have a pointer at index 0 in an array of size arrLen.
At each step, you can move 1 position to the left, 1 position to the right in the array or stay in the same place
 (The pointer should not be placed outside the array at any time).

Given two integers steps and arrLen, return the number of ways such that your pointer still at index 0
after exactly steps steps.

Since the answer may be too large, return it modulo 10^9 + 7.
"""

from functools import lru_cache


class Solution:
    def numWays(self, steps: int, arrLen: int) -> int:
        """
        Link with multi-nomial distributions:
        - p(stay in place) = sum{I=0,S} p(stay in place with S - I steps) * (N choose I) * I!
        - p(stay in place in K) = (K choose K/2)

        But we have to deal with the index 0 and the array length
        """

        M = 1000000007

        @lru_cache(maxsize=None)
        def visit(pos: int, steps: int):
            if pos == 0 and steps == 0:
                return 1
            elif steps == 0:
                return 0

            total = 0
            if pos > 0:
                total += visit(pos - 1, steps - 1)
                total %= M
            if pos < arrLen - 1:
                total += visit(pos + 1, steps - 1)
                total %= M
            total += visit(pos, steps - 1)
            return total % M

        return visit(0, steps)
