from collections import Counter
from functools import lru_cache


"""
https://leetcode.com/problems/scramble-string
"""

class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        """
        Simply testing that both strings contain the same strings does not work:

        Counter example: "abcde" and "caebd".
        "ab" is in the inverted order "ba" but separated by "e" which is before "d"
        => There are two crossed inversions and this is not valid (inversions cut the space)

        Algorithm 1
        -----------
        Search exhaustively for a scrambling of "s1" that matches "s1".

        (?) How do you generate all possibilities without duplicates?
            This repeats the solutions:

        def visit(s):
            if len(s) <= 1:
                return set([s])

            solutions = set()
            for i in range(1, len(s)):
                lefts = visit(s[:i])
                rights = visit(s[i:])
                for l in lefts:
                    for r in rights:
                        solutions.add(l + r)
                        solutions.add(r + l)
            return solutions


        Algorithm 2
        -----------
        Search for a scrambling of part s1[:i] that matches s2[:i] for all i
        """

        def visit(s1, s2):
            n = len(s1)
            m = len(s2)
            if n != m:
                return False

            if s1 == s2:
                return True

            if sorted(s1) != sorted(s2):
                return False

            if n < 4:
                return True

            for i in range(1, n):
                if visit(s1[:i], s2[:i]) and visit(s1[i:], s2[i:]) or visit(s1[:i], s2[-i:]) and visit(s1[i:], s2[:-i]):
                    return True
            return False

        return visit(s1, s2)



