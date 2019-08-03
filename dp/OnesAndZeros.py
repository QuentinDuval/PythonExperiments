"""
In the computer world, use restricted resource you have to generate maximum benefit is what we always want to pursue.

For now, suppose you are a dominator of m 0s and n 1s respectively.
On the other hand, there is an array with strings consisting of only 0s and 1s.

Now your task is to find the maximum number of strings that you can form with given m 0s and n 1s. Each 0 and 1
can be used at most once.

Note:
* The given numbers of 0s and 1s will both not exceed 100
* The size of given string array won't exceed 600.
"""

from collections import defaultdict
from functools import lru_cache
from typing import List


class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        """
        It is unlikely that there is an optimal solution
        => We must explore the solutions

        The recurence is pretty straightforward:
        - either we form the string and recur with less m and n
        - or we do not form the string and recur with same m and n

        Recur(s_pos, m, n)
             = max(1 + Recur(s_pos + 1, m - X, n - Y), Recur(s_pos + 1, m, n))

        => O(len(strs) * m * n) sub-solutions

        We can precompute the number of 1 and 0 in each string to go faster.
        We can group the similar strings in input (that will help).
        => Beats 96%
        """

        counts = defaultdict(int)
        for s in strs:
            zeros = ones = 0
            for c in s:
                if c == '0':
                    zeros += 1
                else:
                    ones += 1
            counts[(zeros, ones)] += 1

        strs = list(counts.keys())
        strs.sort(key=lambda p: p[0] + p[1])

        @lru_cache(maxsize=None)
        def recur(start, m, n):
            if start == len(strs):
                return 0

            zeros, ones = strs[start]
            if m + n < zeros + ones:
                return 0

            max_count = 0
            q = counts[(zeros, ones)]
            if zeros > 0:
                q = min(q, m // zeros)
            if ones > 0:
                q = min(q, n // ones)
            for i in range(q + 1):
                count = i + recur(start + 1, m - i * zeros, n - i * ones)
                max_count = max(max_count, count)
            return max_count

        return recur(0, m, n)

    def other_solution(self, strs: List[str], m: int, n: int) -> int:
        """
        But 29 times slower (beats only 17%)
        Here to top bottom is better...
        """
        dp = [[0] * n for _ in range(m)]
        for s in strs:
            zeros = len(s.replace('1', ''))
            ones = len(s) - zeros
            for i in reversed(range(zeros, m)):
                for j in reversed(range(ones, n)):
                    dp[i][j] = max(dp[i][j], 1 + dp[i-zeros][j-ones])
        return dp[m][n]

