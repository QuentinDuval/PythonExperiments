"""
https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/

Given a string s. In one step you can insert any character at any index of the string.

Return the minimum number of steps to make s palindrome.

A Palindrome String is one that reads the same backward as well as forward.
"""


from functools import lru_cache


class Solution:
    def minInsertions(self, s: str) -> int:

        @lru_cache(maxsize=None)
        def visit(lo: int, hi: int) -> int:
            if lo >= hi:
                return 0
            if s[lo] == s[hi]:
                return visit(lo + 1, hi - 1)
            else:
                return 1 + min(visit(lo, hi - 1), visit(lo + 1, hi))

        return visit(0, len(s) - 1)
