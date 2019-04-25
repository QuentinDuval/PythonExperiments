"""
https://leetcode.com/problems/permutation-in-string

Given two strings s1 and s2, write a function to return true if s2 contains the permutation of s1.
In other words, one of the first string's permutations is the substring of the second string.
"""


class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        """
        To search for the permutation, look for the canonical form (sort s1 or count letters)
        Eliminating letters of s2 not in s1 would not work: violate substring

        Another idea that does not work
        - iterate on 's2', finding areas where we letters are in s1
        - count letters in these areas
        - if superior to count of letters in s1: found it
        Why? Because we cannot skip letters.

        So you have to look at windows!
        - If you sort each time you look at a window: timeout
        - So you have to do a canonical form based on letter count
        """

        l1 = len(s1)
        if l1 == 0:
            return True

        l2 = len(s2)
        if l2 < l1:
            return False

        canonical_form_1 = self.makeHist(s1)
        canonical_form_2 = self.makeHist(s2[:l1])

        if self.checkEqual(canonical_form_1, canonical_form_2):
            return True

        for i in range(l1, l2):
            prev_c = s2[i - l1]
            c = s2[i]
            canonical_form_2[prev_c] = canonical_form_2.get(prev_c, 0) - 1
            canonical_form_2[c] = canonical_form_2.get(c, 0) + 1
            if self.checkEqual(canonical_form_1, canonical_form_2):
                return True

        return False

    def makeHist(self, values):
        hist = {}
        for c in values:
            hist[c] = hist.get(c, 0) + 1
        return hist

    def checkEqual(self, canonical_form_1, canonical_form_2):
        return all(canonical_form_1[c] <= canonical_form_2.get(c, 0) for c in canonical_form_1.keys())
