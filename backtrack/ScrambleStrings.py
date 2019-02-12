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
        The key problem is that there are two many solutions => backtracking with pruning is needed

        (?) How do you generate all possibilities without duplicates? This repeats the solutions:

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

    def isScrambleDP(self, s1: str, s2: str) -> bool:
        """
        Algorithm 3
        -----------
        Dynamic programming
        """

        n = len(s1)
        match = [[[False for k in range(n + 1)] for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                match[i][j][1] = s1[i] == s2[j]

        for k in range(2, n + 1):
            for i in range(n - k + 1):
                for j in range(n - k + 1):
                    match[i][j][k] = False
                    for l in range(1, k):
                        if match[i][j][l] and match[i + l][j + l][k - l]:
                            match[i][j][k] = True
                        elif match[i][j + k - l][l] and match[i + l][j][k - l]:
                            match[i][j][k] = True

        return match[0][0][n]


