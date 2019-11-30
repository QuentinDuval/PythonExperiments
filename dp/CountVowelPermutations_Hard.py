"""
https://leetcode.com/problems/count-vowels-permutation/

Given an integer n, your task is to count how many strings of length n can be formed under the following rules:

Each character is a lower case vowel ('a', 'e', 'i', 'o', 'u')
Each vowel 'a' may only be followed by an 'e'.
Each vowel 'e' may only be followed by an 'a' or an 'i'.
Each vowel 'i' may not be followed by another 'i'.
Each vowel 'o' may only be followed by an 'i' or a 'u'.
Each vowel 'u' may only be followed by an 'a'.

Since the answer may be too large, return it modulo 10^9 + 7.
"""


from functools import lru_cache


class Solution:
    def countVowelPermutation(self, n: int) -> int:
        if n == 0:
            return 0

        """
        Idea is good, but maximum recursion is reached...
        """

        '''
        @lru_cache(maxsize=None)
        def visit(prev: str, pos: int) -> int:
            if pos >= n:
                return 1

            if prev == 'a':
                return visit('e', pos+1)
            elif prev == 'e':
                return sum(visit(c, pos+1) for c in "ai")
            elif prev == 'i':
                return sum(visit(c, pos+1) for c in "aeou")
            elif prev == 'o':
                return sum(visit(c, pos+1) for c in "iu")
            elif prev == 'u':
                return visit('a', pos+1)
            return 0

        mod = 10 ** 9 + 7
        return sum(visit(c, 1) for c in "aeiou") % mod
        '''

        counts = [1] * 5
        for _ in range(n - 1):
            new_counts = [0] * 5
            new_counts[0] = counts[1]
            new_counts[1] = counts[0] + counts[2]
            new_counts[2] = counts[0] + counts[1] + counts[3] + counts[4]
            new_counts[3] = counts[2] + counts[4]
            new_counts[4] = counts[0]
            counts = new_counts
        return sum(counts) % (10 ** 9 + 7)
