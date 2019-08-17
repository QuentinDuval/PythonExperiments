"""
https://leetcode.com/problems/longest-repeating-character-replacement/

Given a string s that consists of only uppercase English letters, you can perform at most k operations on that string.

In one operation, you can choose any character of the string and change it to any other uppercase English character.

Find the length of the longest sub-string containing only repeating letters you can get after
performing the above operations.
"""

from collections import deque, defaultdict


class Solution:
    def characterReplacement_1(self, s: str, k: int) -> int:
        """
        Solution based on 26 passes (one for each letter) of complexity O(N):

        For each letter:
        - extend a window, adding bad characters indexes to a queue
        - when the queue is too big, pop an index from the queue
        - keep track of the longest window

        Complexity is O(N) in time and space, 760 ms, beats 5% only
        """

        letters = set(s)
        longest = 0
        for letter in letters:
            start = 0
            end = 0
            bads = deque()
            while end < len(s):
                if s[end] != letter:
                    bads.append(end)
                if len(bads) > k:
                    start = bads.popleft() + 1
                end += 1
                longest = max(longest, end - start)
        return longest

    def characterReplacement(self, s: str, k: int) -> int:
        """
        Solution based on 1 pass and complexity O(N * number of chars)

        Same window idea but uses a counter of each letter:
        - add letter to a counter of letter in the window
        - if width - max_letter_count > k, reduce the window
        - else extend the window
        - and keep track of longest window

        Complexity is the same, but 188 ms and beats 35%.
        """

        counter = defaultdict(int)
        longest = 0

        start = 0
        end = 0
        while end < len(s):
            counter[s[end]] += 1
            end += 1
            # While is important cause you could delete from the maximum of the counter
            while end - start - max(counter.values()) > k:
                counter[s[start]] -= 1
                start += 1
            longest = max(longest, end - start)
        return longest

