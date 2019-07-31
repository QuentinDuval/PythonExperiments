"""
https://leetcode.com/problems/find-all-anagrams-in-a-string

Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.

Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.

The order of output does not matter.
"""

from collections import defaultdict
from typing import List


class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        starts = []
        n = len(s)
        k = len(p)
        if n < k:
            return starts

        footprint = defaultdict(int)
        for c in p:
            footprint[c] += 1

        window = defaultdict(int)
        for i in range(k):
            window[s[i]] += 1

        if window == footprint:
            starts.append(0)

        for i in range(n - k):
            c = s[i]
            window[c] -= 1
            if window[c] == 0:
                del window[c]
            c = s[i + k]
            window[c] += 1
            if window == footprint:
                starts.append(i + 1)

        return starts



