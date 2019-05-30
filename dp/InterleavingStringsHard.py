from functools import lru_cache


"""
https://leetcode.com/problems/interleaving-string/
Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.
"""


def isInterleave(s1: str, s2: str, s3: str) -> bool:
    """
    Dynamic programming solution in O(n^2)
    --------------------------------------
    The recurrence is straightforward:
    - try consuming a character of s1 if s1 starts with same character as s3
    - try consuming a character of s2 if s2 starts with same character as s3
    => The sub-solutions overlaps so we can use dynamic programming.

    The complexity is proportional to the number of sub-problems.
    How many are they? At most len(s1) * len(s2).
    => The complexity is O(n^2) in time and in space worst case.
    """

    @lru_cache(maxsize=None)
    def valid_interleaving(i1: int, i2: int) -> bool:
        i3 = i1 + i2
        if i1 == len(s1):
            return s2[i2:] == s3[i3:]
        if i2 == len(s2):
            return s1[i1:] == s3[i3:]

        valid = False
        if s1[i1] == s3[i3]:
            valid = valid or valid_interleaving(i1 + 1, i2)
        if s2[i2] == s3[i3]:
            valid = valid or valid_interleaving(i1, i2 + 1)
        return valid

    return valid_interleaving(0, 0)
