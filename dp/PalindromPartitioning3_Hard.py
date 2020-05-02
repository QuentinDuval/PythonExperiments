"""
https://leetcode.com/problems/palindrome-partitioning-iii/

You are given a string s containing lowercase letters and an integer k. You need to :

First, change some characters of s to other lowercase English letters.
Then divide s into k non-empty disjoint substrings such that each substring is palindrome.
Return the minimal number of characters that you need to change to divide the string.
"""

from functools import lru_cache
import numpy as np


class Solution:
    def palindromePartition(self, s: str, k: int) -> int:

        @lru_cache(maxsize=None)
        def to_palindrom(lo: int, hi: int) -> int:
            if lo >= hi:
                return 0

            if s[lo] == s[hi]:
                return to_palindrom(lo + 1, hi - 1)
            else:
                return 1 + to_palindrom(lo + 1, hi - 1)

        @lru_cache(maxsize=None)
        def visit(i: int, k: int) -> int:
            if k <= 0:
                return np.inf
            if k == 1:
                return to_palindrom(i, len(s) - 1)
            if i == len(s):
                return 0 if k == 0 else np.inf

            min_cost = np.inf
            for j in range(i, len(s)):
                min_cost = min(min_cost, to_palindrom(i, j) + visit(j + 1, k - 1))
            return min_cost

        return visit(0, k)
