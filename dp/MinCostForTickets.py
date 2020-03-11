"""
https://leetcode.com/problems/minimum-cost-for-tickets

In a country popular for train travel, you have planned some train travelling one year in advance.
The days of the year that you will travel is given as an array days.  Each day is an integer from 1 to 365.

Train tickets are sold in 3 different ways:

* a 1-day pass is sold for costs[0] dollars;
* a 7-day pass is sold for costs[1] dollars;
* a 30-day pass is sold for costs[2] dollars.

The passes allow that many days of consecutive travel. For example, if we get a 7-day pass on day 2, then we can travel
for 7 days: day 2, 3, 4, 5, 6, 7, and 8.

Return the minimum number of dollars you need to travel every day in the given list of days.
"""

import bisect
from functools import lru_cache
from typing import List


class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        durations = [1, 7, 30]

        @lru_cache(maxsize=None)
        def cost_from(curr_index: int) -> int:
            if curr_index >= len(days):
                return 0

            min_cost = costs[0] * (len(days) - curr_index)
            for duration, cost in zip(durations, costs):
                next_index = bisect.bisect_left(days, days[curr_index] + duration)
                min_cost = min(min_cost, cost + cost_from(next_index))
            return min_cost

        return cost_from(0)
