"""
https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/

You have d dice, and each die has f faces numbered 1, 2, ..., f.

Return the number of possible ways (out of fd total ways) modulo 10^9 + 7 to roll the dice so the sum of
the face up numbers equals target.
"""


from functools import lru_cache


class Solution:
    def numRollsToTarget(self, rolls: int, f: int, target: int) -> int:
        M = 1000000007

        @lru_cache(maxsize=None)
        def visit(rolls: int, target: int) -> int:
            if rolls == 0:
                return target == 0
            if target == 0:
                return rolls == 0

            if rolls * f < target:
                return 0

            total = 0
            for d in range(1, f + 1):
                total += visit(rolls - 1, target - d)
                total %= M
            return total % M

        return visit(rolls, target)
