"""
https://leetcode.com/problems/minimum-window-substring

Given a string S and a string T, find the minimum window in S which will contain all the characters in T
in complexity O(n).
"""

from collections import Counter


# TODO - wonderful article on how you can optimize the code by algorithmic and then play will allocations


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

    def minWindow_3(self, s: str, t: str) -> str:
        """
        Just fuse the map 'window' and 'pattern' => pattern becomes the number of characters to expect
        => Complexity is O(N) and takes 140 ms (nice optimization)
        """
        if not t:
            return ""

        pattern = Counter(t)
        match_count = 0
        min_window = None

        start = 0
        for end in range(len(s)):
            # Key optimization here:
            # You do not need to check the whole pattern, instead just
            # look for the current character and decrease the count of
            # NEEDED characters
            if pattern[s[end]] > 0:
                match_count += 1
            pattern[s[end]] -= 1

            if match_count == len(t):
                # Key optimisation here:
                # You do not need to check the whole pattern, instead just
                # look for a character whose NEEDED characters is 0
                while pattern[s[start]] < 0:
                    pattern[s[start]] += 1
                    start += 1

                if not min_window or min_window[1] - min_window[0] > end - start:
                    min_window = start, end

                # Move past the current window (look for another one)
                pattern[s[start]] += 1
                match_count -= 1
                start += 1

        return s[min_window[0]:min_window[1] + 1] if min_window else ""

    def minWindow_3(self, s: str, t: str) -> str:
        """
        Just use plain maps and not Counter
        => Complexity is O(N) and takes 108 ms (nice optimization)
        """
        if not t:
            return ""

        pattern = {}
        for c in t:
            pattern[c] = pattern.get(c, 0) + 1
        match_count = 0
        min_window = None

        start = 0
        for end in range(len(s)):
            # Key optimization here:
            # You do not need to check the whole pattern, instead just
            # look for the current character and increase the count of
            # matched characters
            if pattern.get(s[end], 0) > 0:
                match_count += 1
            pattern[s[end]] = pattern.get(s[end], 0) - 1

            if match_count == len(t):
                # Key optimisation here:
                # You do not need to check the whole pattern, instead just
                # look for a character whose count is == pattern count
                while pattern[s[start]] < 0:
                    pattern[s[start]] += 1
                    start += 1

                if not min_window or min_window[1] - min_window[0] > end - start:
                    min_window = start, end

                # Move past the current window (look for another one)
                pattern[s[start]] += 1
                match_count -= 1
                start += 1

        return s[min_window[0]:min_window[1] + 1] if min_window else ""

    def minWindow_4(self, s: str, t: str) -> str:
        """
        Avoid allocating memory / modifying the map for characters not in the pattern 't'
        => Complexity is O(N) and takes 84 ms (nice optimization)
        """

        if not t:
            return ""

        pattern = {}
        for c in t:
            pattern[c] = pattern.get(c, 0) + 1
        match_count = 0
        min_window = None

        start = 0
        for end in range(len(s)):
            # Key optimization here:
            # You do not need to check the whole pattern, instead just
            # look for the current character and increase the count of
            # matched characters
            count = pattern.get(s[end])
            if count is not None:
                if count > 0:
                    match_count += 1
                pattern[s[end]] = count - 1

            if match_count == len(t):
                # Key optimisation here:
                # You do not need to check the whole pattern, instead just
                # look for a character whose count is == pattern count
                while start < end:
                    count = pattern.get(s[start])
                    if count is None:
                        start += 1
                    elif count < 0:
                        pattern[s[start]] = count + 1
                        start += 1
                    else:
                        break

                if not min_window or min_window[1] - min_window[0] > end - start:
                    min_window = start, end

                # Move past the current window (look for another one)
                pattern[s[start]] += 1
                match_count -= 1
                start += 1

        return s[min_window[0]:min_window[1] + 1] if min_window else ""
