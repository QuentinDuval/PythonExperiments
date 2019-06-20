"""
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/

Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like
(ie, buy one and sell one share of the stock multiple times) with the following restrictions:

- You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
- After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)
"""

from typing import List


from functools import lru_cache


class Solution:
    def maxProfit_dp(self, prices: List[int]) -> int:
        """
        Divide the problem:
        - If you select to buy and sell in [i,j] => sub-problem for [j+2, n]
        - Find the maximum you can do by trying systematically all possibilities
        => Complexity is in the O(N**2) maximum
        """

        @lru_cache(maxsize=len(prices))
        def maximize(i: int) -> int:
            if i >= len(prices):
                return 0

            max_profit = 0
            min_val = prices[i]
            max_val = prices[i]

            for j in range(i + 1, len(prices)):

                # Optimization: no need to keep the previous buy point if it was higher
                if prices[j] < min_val:
                    min_val = prices[j]
                    max_val = prices[j]

                # Optimization: no need to check for solution after if it does not improve the situation
                elif prices[j] > max_val:
                    max_val = prices[j]
                    max_profit = max(max_profit, prices[j] - min_val + maximize(j + 2))

            return max_profit

        return maximize(0)

    def maxProfit(self, prices: List[int]) -> int:
        """
        You can actually do it in one pass:
        https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/discuss/300675/Java-solution-O(n)-time-%2B-explanation
        """
        if not prices:
            return 0

        max_profit_buy = -prices[0]  # Max profit after having bought
        max_profit = 0  # Max profit after having sold
        max_profit_before = 0  # The previous iteration of max_profit (to take into account delay)

        for price in prices[1:]:
            max_profit_buy = max(max_profit_buy, max_profit_before - price)
            max_profit_before = max_profit
            max_profit = max(max_profit, max_profit_buy + price)

        return max_profit



