"""
https://leetcode.com/problems/guess-number-higher-or-lower-ii/

We are playing the Guess Game. The game is as follows:

I pick a number from 1 to n. You have to guess which number I picked.

Every time you guess wrong, I'll tell you whether the number I picked is higher or lower.

However, when you guess a particular number x, and you guess wrong, you pay $x.
You win the game when you guess the number I picked.
"""


class Solution:
    def getMoneyAmount(self, n: int) -> int:
        """
        You have to pick a number such that each side divides equally
        (same number of money paid in worst case if lower or higher)

        So we want to minimimize this (for all i):
        i + max(getMoneyAmount(1, i-1), getMoneyAmount(i+1, n))
        """

        def cache(f):
            memo = {}

            def wrapped(*args):
                if args in memo:
                    return memo[args]
                res = f(*args)
                memo[args] = res
                return res

            return wrapped

        @cache
        def minCost(lo: int, hi: int) -> int:
            if lo >= hi:
                return 0
            if lo == hi - 1:
                return lo

            res = float('inf')
            mid = lo + (hi - lo) // 2
            for i in range(mid, hi):
                res = min(res, i + max(minCost(lo, i - 1), minCost(i + 1, hi)))
            return res

        return minCost(1, n)

