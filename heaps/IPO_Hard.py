"""
https://leetcode.com/problems/ipo/

Suppose LeetCode will start its IPO soon. In order to sell a good price of its shares to Venture Capital,
LeetCode would like to work on some projects to increase its capital before the IPO. Since it has limited resources,
it can only finish at most k distinct projects before the IPO. Help LeetCode design the best way to maximize its total
capital after finishing at most k distinct projects.

You are given several projects. For each project i, it has a pure profit Pi and a minimum capital of Ci is needed to
start the corresponding project. Initially, you have W capital. When you finish a project, you will obtain its
pure profit and the profit will be added to your total capital.

To sum up, pick a list of at most k distinct projects from given projects to maximize your final capital,
and output your final maximized capital.
"""

import heapq
from typing import List


class Solution:
    def findMaximizedCapital(self, k: int, initial_capital: int,
                             profits: List[int], capitals: List[int]) -> int:
        """
        Just invest the capital needed to make as much money as possible
        Keep track of which investment you can do
        (put in a priority queue only profits that are possible)

        Complexity is O(N log N) and beats 79%
        """

        # Sort the investments by capital
        n = len(profits)
        indexes = list(range(n))
        indexes.sort(key=lambda i: capitals[i])

        # Integrate the investments requires less than capital
        def add_profit_up_to(max_capital: int, last_index: int):
            while last_index < len(indexes) and capitals[indexes[last_index]] <= max_capital:
                heapq.heappush(possible_profits, -1 * profits[indexes[last_index]])
                last_index += 1
            return last_index

        # First add the investments below the initial capital
        possible_profits = []
        last_index = add_profit_up_to(initial_capital, 0)

        # For as long as we can make some investments
        current_capital = initial_capital
        for _ in range(k):
            if not possible_profits:
                break

            # Make the highest profit possible
            current_capital += heapq.heappop(possible_profits) * -1

            # Add new eligible investments to the pool
            last_index = add_profit_up_to(current_capital, last_index)

        return current_capital
