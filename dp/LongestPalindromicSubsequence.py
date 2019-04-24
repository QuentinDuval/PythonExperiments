from functools import *


class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        """
        Recurrence relation, starting from beginning and end of string:
        - Either the letters are the same => 2 + recurse
        - Either the letters are different => try dropping on both sides

        Use dynamic programming to make sure you do not evaluate the same relation twice

        Number of sub-solutions: O(N**2)
        - One for each start
        - One for each end > start

        Time complexity: O(N**2)
        Memory complexity: O(N) if you are clever (try with solution of length 1, then 2, then 3, etc.)
        """

        '''
        # Solution with O(N**2) space
        @lru_cache(maxsize=None)
        def visit(start, end):
            if end < start:
                return 0
            if start == end:
                return 1

            if s[start] == s[end]:
                return 2 + visit(start+1, end-1)
            else:
                return max(visit(start+1, end), visit(start, end-1))

        return visit(0, len(s)-1)
        '''

        # Solution using O(n) space

        n = len(s)
        prev_memo = [0] * n
        memo = [1] * n

        for l in range(2, n + 1):
            new_memo = [0] * n
            for start in range(0, n - l + 1):
                end = start + l - 1
                if s[start] == s[end]:
                    new_memo[start] = 2 + prev_memo[start + 1]
                else:
                    new_memo[start] = max(memo[start + 1], memo[start])
            prev_memo, memo = memo, new_memo

        return memo[0]


