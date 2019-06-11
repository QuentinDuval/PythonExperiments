"""
Given a string s, partition s such that every substring of the partition is a palindrome.
Return all possible palindrome partitioning of s.
"""


from typing import List


class Solution:
    def partition_dp(self, s: str) -> List[List[str]]:
        """
        You can compute for all i < j if s[i:j+1] is a palindrom in O(N**2) using DP.

        Then you can visit from the left, trying all palindrome starting at i:
        There are overlapping sub-problems (selecting palindrom size 2 then 3 and conversly)

        All DP leads to a complexity of O(N**2)
        """

        def cache(f):
            memo = {}
            def wrapped(*args):
                res = memo.get(args)
                if res is not None: return res
                res = f(*args)
                memo[args] = res
                return res
            return wrapped

        @cache
        def is_palindrom(i: int, j: int):
            if i == j:
                return True
            if s[i] != s[j]:
                return False
            if s[i] == s[j]:
                return i == j + 1 or is_palindrom(i + 1, j - 1)

        @cache
        def visit(i: int) -> List[List[int]]:
            if i == len(s):
                return [[]]

            solutions = []
            for j in range(i, len(s)):
                if is_palindrom(i, j):
                    for sol in visit(j + 1):
                        solutions.append([s[i:j + 1]] + sol)
            return solutions

        return visit(0)

    def partition(self, s: str) -> List[List[str]]:
        """
        Solution based on the observation that on a new letter:
        - we could just add the letter as a new partition
        - the letter could make the last palindrome a palindrome
        - the letter could make the two last palindrom a palindrome

        Goes a bit faster in practice.
        """

        # TODO - estimate complexity

        def isPalindrom(s):
            i = 0
            j = len(s) - 1
            while i < j:
                if s[i] == s[j]:
                    i += 1
                    j -= 1
                else:
                    return False
            return True

        if not s:
            return []

        solutions = [[s[0]]]
        for c in s[1:]:
            n = len(solutions)
            for i in range(n):
                solution = solutions[i]
                if isPalindrom(solution[-1] + c):
                    solutions.append(solution[:-1] + [solution[-1] + c])
                elif len(solution) > 1 and solution[-2] == c:
                    solutions.append(solution[:-2] + [solution[-2] + solution[-1] + c])
                solution.append(c)
        return solutions

