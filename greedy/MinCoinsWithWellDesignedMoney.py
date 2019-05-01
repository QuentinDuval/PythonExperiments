"""
https://practice.geeksforgeeks.org/problems/-minimum-number-of-coins/0

When the money is well designed (each bigger bill can be decomposed in smaller bills),
we can find the best coins to use by just using a greedy strategy.
"""


available_coins = [1, 2, 5, 10, 20, 50, 100, 200, 500, 2000]


def min_coins(amount):
    coins = []
    for available in reversed(available_coins):
        if amount >= available:
            q, amount = divmod(amount, available)
            coins.extend(available for _ in range(q))
    return coins
