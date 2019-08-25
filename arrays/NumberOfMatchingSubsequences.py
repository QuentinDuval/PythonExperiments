"""
https://leetcode.com/problems/number-of-matching-subsequences/

Given string S and a dictionary of words words, find the number of words[i] that is a subsequence of S.
"""


from typing import List


class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        """
        Brute force would be to try all words and check if they are a subsequence of S.
        The complexity would be O(len(s) * len(words) * len(word))

        Ideally, we can go down to O(len(words) * len(word)) by having a kind of TRIE:
        - for each character of S, have the index of next characters after
        - then for each word of words, just follow the pointers
        => requires an array of O(len(s) * 26) indexes, which can be constructed right to left
        """

        # At position 'i': unmatched character s[i] gets mapped to 'i+1' (next cell)
        skips = [[-1] * 26 for _ in range(len(s) + 1)]
        for i in reversed(range(len(s))):
            c = s[i]
            skips[i][:] = skips[i + 1]
            skips[i][ord(c) - ord('a')] = i + 1

        nb_subseq = 0
        for word in words:
            state = 0
            for c in word:
                state = skips[state][ord(c) - ord('a')]
                if state == -1:
                    break
            if state != -1:
                nb_subseq += 1
        return nb_subseq

