from typing import *


"""
As many transactions as you like
"""


"""
A single transaction
"""


"""
2 transactions
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
"""


class Solution2Transactions:
    def maxProfit(self, prices: List[int]) -> int:
        """
        The general idea is to find where to split the problem in two.
        Knowing that based on the previous problem, we can solve 1 transaction in O(n).

        The brute force way is to try every point and compute the left and right max profit.
        This approach leads to O(n ^ 2) algorithm. We can do better.

        Our goal would be to pre-compute two tables 'left' and 'right' such that at index 'i':
        - 'left[i]' contains the max profit we can do on the problem prices[0:i]
        - 'right[i]' contains the max profit we can do on the problem prices[i:]

        How to do it?

        We just run the algorithm for 1 transaction left to right and keep in 'left'
        the result of each iteration.

        We then run the same algorithm for 1 transaction from right to left and keep
        in 'right' the result of the iteration.

        We cannot simply run the algorithm for 'left' to get 'right' we reverse collection
        for it inverses the buy and sell.
        """
        if not prices:
            return 0

        n = len(prices)
        left = self.max_left(prices)
        right = self.max_right(prices)
        return max(left[i] + right[i] for i in range(n))

    def max_left(self, prices: List[int]) -> List[int]:
        max_profits = [0]
        min_price = prices[0]
        for price in prices[1:]:
            if price < min_price:
                min_price = price
            max_profits.append(max(max_profits[-1], price - min_price))
        return max_profits

    def max_right(self, prices: List[int]) -> List[int]:
        max_profits = [0]
        max_price = prices[-1]
        for price in prices[:-1][::-1]:
            if price > max_price:
                max_price = price
            max_profits.append(max(max_profits[-1], max_price - price))
        return max_profits[::-1]


"""
K transactions
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/
"""
