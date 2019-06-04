"""
https://leetcode.com/problems/repeated-string-match/

Given two strings A and B, find the minimum number of times A has to be repeated such that B is a substring of it.
If no such solution, return -1.

Note:
The length of A and B will be between 1 and 10000.
"""


class Solution:
    def repeatedStringMatch(self, A: str, B: str) -> int:
        """
        If A has to be repeated to get B as a substring, then we must have:
            B = S + A * N + P
        Where:
            - S is a suffix of A
            - P is a prefix of A

        The goal is therefore to find the first occurrence of A in B (if this occurrence is not
        in the first len(A) characters, it cannot be done) and then check that all following
        characters are in A (and cycle through this). Also, the prefix must be a suffix of A.

        Edges cases:
        - B might be smaller than A
        - if B in smaller than A, we might require two As to find a B inside it
        """

        # If len(B) < len(A), it requires special treatment
        if len(B) < len(A):
            if B in A:
                return 1
            if B in A * 2:
                return 2
            return -1

        # Different character sets => no solution possible
        if set(A) != set(B):
            return -1

        # Find the starting position of A in B
        start = None
        for i in range(len(A)):
            valid_start = True
            for j in range(min(len(A), len(B) - i)):
                if A[j] != B[i + j]:
                    valid_start = False
            if valid_start:
                start = i
                break

        count = 0
        if start is None:
            return -1
        elif start > 0:
            count += 1

        if not A.endswith(B[:start]):
            return -1

        while start < len(B) - len(A) + 1:
            if not B[start:].startswith(A):
                return -1
            start += len(A)
            count += 1

        if not A.startswith(B[start:]):
            return -1
        elif start != len(B):
            count += 1
        return count
