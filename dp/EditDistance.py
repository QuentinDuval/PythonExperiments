"""
https://leetcode.com/problems/edit-distance

Given two words word1 and word2, find the minimum number of operations required to convert word1 to word2.

You have the following 3 operations permitted on a word:
* Insert a character
* Delete a character
* Replace a character
"""


from functools import lru_cache


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        """
        It is enough to try to match from left to right:
        the right to left will be managed by deletion / replacement at the end.
        """

        @lru_cache(maxsize=None)
        def distance(start1, start2):
            if start1 == len(word1):
                return len(word2) - start2

            if start2 == len(word2):
                return len(word1) - start1

            if word1[start1] == word2[start2]:
                return distance(start1 + 1, start2 + 1)

            return 1 + min([
                distance(start1 + 1, start2),
                distance(start1, start2 + 1),
                distance(start1 + 1, start2 + 1)
            ])

        return distance(0, 0)
