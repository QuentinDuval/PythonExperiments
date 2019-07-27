"""
https://leetcode.com/problems/longest-well-performing-interval/

We are given hours, a list of the number of hours worked per day for a given employee.

A day is considered to be a tiring day if and only if the number of hours worked is (strictly) greater than 8.

A well-performing interval is an interval of days for which the number of tiring days is strictly larger than the number of non-tiring days.

Return the length of the longest well-performing interval.
"""


from typing import List


class Solution:
    def longestWPI(self, hours: List[int]) -> int:
        """
        Looks like a balancing parenthesis problem, but without the ordering
        Much like finding the longest interval with more 1 than 0

        When dealing with problem like this:
        - consider doing a scan from left to right
        - consider doing a cumulative sum (by mapping 1 to 1, 1 to -1)
        - consider using a map (or a stack?) to find the matching value on the left

        Then TRY AN EXAMPLE to better understand what you need:

        Example:     [9, 9, 6, 0, 6, 6, 9, 9]
        Prefix:   [0, 1, 2, 1, 0, -1, -2, -1, 0]
        Map:      {0: -1, 1: 0, 2: 1, -1: 4, -2: 5}

        Then note that:
        * 1 is not useful: if you are superior to 1, you are superior to 0, and thus can start at -1
        * -1 is useful: it is after 0, so if 0 is not available, you might need to consider it
        => So we only need to add values that are lesser than previous (and so we can binary search)
        => If fact, we will systematically have all numbers (0, -1, -2 .. -k) so we can index in O(1)
        """
        max_len = 0
        previous = [(0, -1)]
        cumulative = 0

        for i, hour in enumerate(hours):
            cumulative += 1 if hour > 8 else -1
            if previous[-1][0] > cumulative:
                previous.append((cumulative, i))

            previous_index = abs(cumulative-1) if cumulative <= 0 else 0
            if previous_index < len(previous):
                interval_len = i - previous[previous_index][1]
                max_len = max(max_len, interval_len)

        return max_len

        # TODO - do better (only beats 28 %)
