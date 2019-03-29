from typing import List


"""
https://leetcode.com/problems/daily-temperatures/

Given a list of daily temperatures T, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature.
If there is no future day for which this is possible, put 0 instead.

For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].

Note: The length of temperatures will be in the range [1, 30000]. Each temperature will be an integer in the range [30, 100].
"""


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """
        Idea:
        - Start from the end and scan right to left

        Keeping the largest element found so far (and its index) is not good enough: there might be a closer element big enough
        But you could use a STACK: pop from the stack when the element at the top is lower... (current element will be a better match)
        """
        if not temperatures:
            return []

        waiting_times = [0]

        biggests = [(temperatures[-1], len(temperatures) - 1)]
        for i in reversed(range(len(temperatures) - 1)):
            temperature = temperatures[i]
            while biggests and biggests[-1][0] <= temperature:
                biggests.pop()

            if biggests:
                waiting_times.append(biggests[-1][1] - i)
            else:
                waiting_times.append(0)

            biggests.append((temperature, i))

        return waiting_times[::-1]
