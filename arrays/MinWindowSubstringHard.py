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
        characters in 't' and takes 600 ms
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

            # Note that at this point, we move start past the preceding point

        return s[min_window[0]:min_window[1] + 1] if min_window else ""

    def minWindow_2(self, s: str, t: str) -> str:
        """
        Instead of looking for the full pattern when incrementing 'start', you can just look at the letter
        in the pattern and the window and stop when their count is equal
        => Complexity falls to O(N) and takes 160 ms (massive optimization)
        """

        if not t:
            return ""

        pattern = Counter(t)
        window = Counter()

        match_count = 0
        min_window = None

        start = 0
        for end in range(len(s)):
            # Key optimization here:
            # You do not need to check the whole pattern, instead just
            # look for the current character and increase the count of
            # matched characters
            if window[s[end]] < pattern[s[end]]:
                match_count += 1
            window[s[end]] += 1

            if match_count == len(t):
                # Key optimisation here:
                # You do not need to check the whole pattern, instead just
                # look for a character whose count is == pattern count
                while window[s[start]] > pattern[s[start]]:
                    window[s[start]] -= 1
                    start += 1

                if not min_window or min_window[1] - min_window[0] > end - start:
                    min_window = start, end

                # Move past the current window (look for another one)
                window[s[start]] -= 1
                match_count -= 1
                start += 1

        return s[min_window[0]:min_window[1] + 1] if min_window else ""

