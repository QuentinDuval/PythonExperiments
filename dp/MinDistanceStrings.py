"""
https://leetcode.com/problems/delete-operation-for-two-strings
"""


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        """
        Recurrence is:

        if word1[0] == word2[0]:
            return minDistance(word1[1:], word2[1:])
        else:
            return 1 + min(minDistance(word1[1:], word2), minDistance(word1, word2[1:]))

        Number of sub-solutions: O(N**2)
        - N for each start of word1
        - N for each start of word2

        Complexity: O(N**2) time and O(N**2) space if top-down recursion

        But we can do it in O(N) space if we observe that:
        - We only need the previous word1[1:]
        - We can compute the word2 from the end
        """

        l1 = len(word1)
        l2 = len(word2)
        memo = list(reversed(range(l2 + 1)))

        for i1 in reversed(range(l1)):
            new_memo = [0] * (l2 + 1)
            new_memo[-1] = l1 - i1
            for i2 in reversed(range(l2)):
                if word1[i1] == word2[i2]:
                    new_memo[i2] = memo[i2 + 1]
                else:
                    new_memo[i2] = 1 + min(memo[i2], new_memo[i2 + 1])
            memo = new_memo

        return memo[0]

