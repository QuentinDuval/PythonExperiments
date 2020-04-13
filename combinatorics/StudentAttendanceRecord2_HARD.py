"""
https://leetcode.com/problems/student-attendance-record-ii

Given a positive integer n, return the number of all possible attendance records with length n, which will be regarded as rewardable. The answer may be very large, return it after mod 109 + 7.

A student attendance record is a string that only contains the following three characters:

'A' : Absent.
'L' : Late.
'P' : Present.
A record is regarded as rewardable if it doesn't contain more than one 'A' (absent) or more than two continuous 'L' (late).
"""

from functools import lru_cache


class Solution:
    def checkRecord(self, n: int) -> int:
        """
        - No more than 1 absent
        - No more than 2 contiguous late
        - All the other must be present

        IDEA 1:
        Split the problem in two: without absent for n + n * without absent for n-1
        => Not working, because would split the 'L'

        IDEA 2:
        Split the problem in two:
        * without 'A'
        * with 'A': cut the problem in two in that place
        """

        M = 1000000007
        MAX_LATE = 2

        # TOO HEAVY
        '''
        @lru_cache(maxsize=None)
        def without_absent(n: int, prev_is_L: bool):
            if n < 0: return 0
            if n == 0: return 1

            count = 1
            start = 1 if prev_is_L else 0
            for i in range(start, n+1):
                for l in range(1, MAX_LATE+1):
                    remaining = n-i-l
                    if remaining >= 0:
                        sub_count = without_absent(remaining, True)
                        count = (count + sub_count) % M
            return count % M

        count = without_absent(n, False)
        for i in range(n):
            sub_count = without_absent(i, False) * without_absent(n-i-1, False)
            count = (count + sub_count) % M
        return count % M
        '''

        # Much better, realize there are only 3 possibilities:
        # - no L at this position, skip 1
        # - one L at this position, skip 2
        # - two L at this position, skip 3
        # => unfortunately, depth recursion exceeded
        '''
        @lru_cache(maxsize=None)
        def without_absent(n: int):
            if n < 0: return 0
            if n == 0: return 1
            if n == 1: return 2
            if n == 2: return 4
            if n == 3: return 7

            count = without_absent(n-1)
            count += without_absent(n-2)
            count += without_absent(n-3)
            return count % M

        count = without_absent(n)
        for i in range(n):
            sub_count = without_absent(i) * without_absent(n-i-1)
            count = (count + sub_count) % M
        return count % M
        '''

        memo = [0] * (n + 1)
        memo[:4] = [1, 2, 4, 7]
        for i in range(4, n + 1):
            memo[i] = (memo[i - 1] + memo[i - 2] + memo[i - 3]) % M

        count = memo[n]
        for i in range(n):
            count = (count + memo[i] * memo[n - i - 1]) % M
        return count

