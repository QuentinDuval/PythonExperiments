"""
https://leetcode.com/problems/minimum-window-substring

Given a string S and a string T, find the minimum window in S which will contain all the characters in T
in complexity O(n).
"""

from collections import Counter


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        """
        Solution (Window):
        The brute force solution would consist in trying all i < j
        and check if it contains the same number of characters
        => Complexity is therefore O(N * N * M)

        But trying all i < j is not useful:
        - for a working i < j, we want to augment i to reduce the size
        - for a working i < j, we want to decrease j to reduce the size

        So we can use a window [i, j] and expand j until we find a match,
        then decrease i while there is a match... and continue.

        We can use a canonical form for our strings (counter, a map) to check
        the inclusion.

        => Complexity is therefore O(N * K) where is the number of different
        characters in 't' and it beats only 8%
        """

        if not t:
            return ""

        pattern = Counter(t)
        window = Counter()

        def contains(window, pattern):
            for c, count in pattern.items():
                if window[c] < count:
                    return False
            return True

        min_window = None

        start = 0
        for end in range(len(s)):
            window[s[end]] += 1
            if not contains(window, pattern):
                continue

            while contains(window, pattern):
                window[s[start]] -= 1
                start += 1

            if not min_window or min_window[1] - min_window[0] > end - start + 1:
                min_window = start - 1, end

        if min_window:
            return s[min_window[0]:min_window[1] + 1]
        return ""

