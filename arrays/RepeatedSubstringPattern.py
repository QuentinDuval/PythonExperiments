"""
EASY
https://leetcode.com/problems/repeated-substring-pattern

Given a non-empty string check if it can be constructed by taking a substring of it and appending multiple copies of the substring together.
You may assume the given string consists of lowercase English letters only and its length will not exceed 10000.
"""


class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        """
        The brute force way is to try all combinations of i < j.

        But some important observations reduce this:
        - If a string is repeated by a given pattern, this pattern starts at the beginning
        - This pattern must end with the last letter of s
        - This pattern cannot be bigger than half the string
        - The length of this pattern must divide exactly the length of the string
        """
        n = len(s)
        if n < 2:
            return False

        last_letter = s[-1]
        for i in range(1 + n // 2):
            if s[i] == last_letter:
                q, r = divmod(n, i + 1)
                if r == 0 and q > 1:
                    if s == s[:i + 1] * q:
                        return True
        return False
