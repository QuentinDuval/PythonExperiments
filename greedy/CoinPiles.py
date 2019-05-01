"""
https://practice.geeksforgeeks.org/problems/coin-piles/0

There are N piles of coins each containing  Ai (1<=i<=N) coins.

Now, you have to adjust the number of coins in each pile such that for any two pile, if a be the number of coins in
first pile and b is the number of coins in second pile then |a-b|<=K.

In order to do that you can remove coins from different piles to decrease the number of coins in those piles but you
cannot increase the number of coins in a pile by adding more coins.

Now, given a value of N and K, along with the sizes of the N different piles you have to tell the minimum number of
coins to be removed in order to satisfy the given condition.

Note: You can also remove a pile by removing all the coins of that pile.
"""

from functools import lru_cache


"""
The following algorithm does not work.
Indeed, we can also remove the coin pile entirely:
1 5 5 5 5 and k = 3 should return 1 
"""


def min_removal_bad(heights, k):
    if not heights:
        return 0

    removals = 0
    lowest = min(heights)
    for height in heights:
        if height > lowest + k:
            removals += height - (lowest + k)
    return removals


"""
The following algorithm work by Dynamic Programming:
- We sort the piles from left minimum to right maximum
- Either we keep the left pile and we remove from the right pile
- Or we drop the left pile entirely and keep the right pile

This works because we cannot add coins to the left pile

Number of sub-solutions: O(N^2)
=> Space complexity: O(N^2)
=> Time complexity: O(N^2) (constant amount of work by recursion)
"""


def min_removal_dp(heights, k):
    if not heights:
        return 0

    heights.sort()

    @lru_cache(maxsize=None)
    def visit(i, j):
        if i >= j or heights[j] - heights[i] <= k:
            return 0

        drop_left = heights[i] + visit(i + 1, j)
        pop_right = visit(i, j - 1) + heights[j] - (heights[i] + k)
        return min(drop_left, pop_right)

    return visit(0, len(heights) - 1)


"""
DP version from bottom-up.

visit(i, j) depends on:
- visit(i+1, j) recurse on solutions of len - 1
- visit(i, j-1) recurse on solutions of len - 1

So you can visit by increasing delta between i and j
=> Space complexity decreased to O(N)
"""


def min_removal_dp2(heights, k):
    if not heights:
        return 0

    heights.sort()

    n = len(heights)
    memo = [0] * n
    for delta in range(1, n):
        new_memo = [0] * n
        for i in range(n - delta):
            j = i + delta
            if heights[j] - heights[i] <= k:
                new_memo[i] = 0
            else:
                drop_left = heights[i] + memo[i+1]
                pop_right = memo[i] + heights[j] - (heights[i] + k)
                new_memo[i] = min(drop_left, pop_right)
        memo = new_memo
    return memo[0]


"""
But you can also do this problem in a greedy fashion
[1, 2, 5, 5, 5, 5], k = 3

First idea (WRONG):
- compare the cost of dropping lowest value (here 1) versus the cost of adapting the next values
- the first time you do not drop, there is no reason to drop anymore (the minimum is still there...)
BUT THIS DOES NOT WORK: you have to keep trying for every start point

Time complexity is O(N**2)
Space complexity is O(1)
"""


# TODO - find explanation of why early cutting does not work...


def min_removal(heights, k):
    if not heights:
        return 0

    heights.sort()

    def remove_top_cost(val, heights):
        cost = 0
        cutoff = val + k
        for h in reversed(heights):
            if h <= cutoff:
                break
            cost += h - cutoff
        return cost

    min_cost = remove_top_cost(heights[0], heights)
    drop_cost = 0
    for start in range(1, len(heights)):
        drop_cost += heights[start-1]
        right_removals = remove_top_cost(heights[start], heights[start:])
        min_cost = min(min_cost, drop_cost + right_removals)
    return min_cost

