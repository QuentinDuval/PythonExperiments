"""
https://leetcode.com/problems/shopping-offers/

In LeetCode Store, there are some kinds of items to sell. Each item has a price.

However, there are some special offers, and a special offer consists of one or more different kinds of items with a sale price.

You are given the each item's price, a set of special offers, and the number we need to buy for each item.
The job is to output the lowest price you have to pay for exactly certain items as given,
where you could make optimal use of the special offers.

Each special offer is represented in the form of an array, the last number represents the price you need to pay for this
special offer, other numbers represents how many specific items you could get if you buy this offer.

You could use any of special offers as many times as you want.
"""

from functools import *
from typing import List


class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        """
        You are not allowed to by more items that necessary
        => This will help in the search to cut branches

        We can compute the base price by just doing a dot product between 'price' and 'needs'
        => This could help in the search to cut branches

        We could do a search offer by offer, but:
        - there are 100 offers (the search tree is big)
        - a lot of sub-problems actually overlap (example of offer [1,2,x] and [2,1,y])

        We can rely on some tricks to help with the search:
        - Pruning:
            - Handle the offers in a specific order
            - Do not consider cases in which the items drop below 0
        - Memoization:
            - Even if we consider the offers in order, there are overlapping sub-solutions
        - Guided search:
            - Try to search where the benefits are the highest
        """

        n = len(needs)

        # Prioritise offers that remove the most items from the needs (and the free offers)
        special.sort(key=lambda offer: -1 * sum(offer[:-1]) if offer[-1] else -float('inf'))

        def all_possible_counts(needs, offer):
            max_offer = min(needs[i] // offer[i] for i in range(n) if offer[i])
            for count in range(max_offer + 1):
                remaining = tuple(needs[i] - count * offer[i] for i in range(n))
                yield offer[-1] * count, remaining

        @lru_cache(maxsize=None)
        def visit(i, needs):
            if i >= len(special):
                return sum(price[i] * needs[i] for i in range(n))

            min_price = float('inf')
            for cost, remaining in all_possible_counts(needs, special[i]):
                min_price = min(min_price, cost + visit(i + 1, remaining))
            return min_price

        return visit(0, tuple(needs))
