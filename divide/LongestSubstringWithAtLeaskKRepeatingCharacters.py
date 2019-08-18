"""
https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/

Find the length of the longest substring T of a given string (consists of lowercase letters only)
such that every character in T appears no less than k times.
"""


from collections import Counter


class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        """
        Window system does not work great... when do you need to reduce it?

        We can easily get a O(N ** 2) algorithm:
        - for each i, try to increase j
        - keep a count of the characters

        But there is a O(N) algorithm:
        - simply count the characters
        - split on every character whose count is lower than 'k'
        - recurse (or if no recusion, you just found a compliant string)
        """

        counter = Counter(s)
        for c, count in counter.items():
            if count < k:
                return max(self.longestSubstring(sub, k) for sub in s.split(c))
        return len(s)
