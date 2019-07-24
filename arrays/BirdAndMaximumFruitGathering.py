"""
https://practice.geeksforgeeks.org/problems/bird-and-maximum-fruit-gathering/0

There are N trees in a circle. Each tree has a fruit value associated with it.
A bird has to sit on a tree for 0.5 sec to gather all the fruits present on the tree and then the bird can move to a neighboring tree.
It takes the bird 0.5 seconds to move from one tree to another.
Once all the fruits are picked from a particular tree, she can’t pick any more fruits from that tree.
The maximum number of fruits she can gather is infinite.

Given N and M (the total number of seconds the bird has), and the fruit values of the trees. We have to maximize the
total fruit value that the bird can gather. The bird can start from any tree.
"""


"""
!!! THE SOLUTION BELOW IS FOR THE CASE THE BIRD CAN SKIP TREES !!!
This is actually not the case in the problem...

Observations:

There is no reason to go right only to go left after:
If so, you should just start from the left and go further
=> There is no reason to chose a different way of moving than
   from left to right (all equivalent).

Easy cases:
M >= N => collect everything
M == 1 => collect the best

A bird can choose to skip a tree.

Brute force algorithm:
- try for every starting point 'i' (first point to be consumed)
- explore each choice "consume or not consume" and recurse
- sub-problems are clearly overlapping here

Recurrence formula is:

SOL(start=i, time=t)
    = max(
        tree[i] + SOL(start=i+1, time=t-1), # collect fruit
        SOL(start=i+1, time=t-0.5)
        )

If we do dynamic programming:
- the number of sub-solutions is O(N * M)
- the cost of recursion is O(1)
=> Complexity is O(N * M) for one starting point
=> Complexity is O(N * N * M) for every starting point

But actually, the sub-solutions should overlap anyway
=> Complexity is O(N * M)
"""


def cached(f):
    memo = {}
    def wrapped(*args):
        if args in memo:
            return memo[args]
        res = f(*args)
        memo[args] = res
        return res
    return wrapped


def maximize_gathering(nums, starting_fuel):
    if not nums:
        return 0

    if starting_fuel == 1:
        return max(nums)

    if starting_fuel >= 2 * len(nums) - 1:
        return sum(nums)

    @cached
    def recur(start, fuel):
        while start >= len(nums):
            start -= len(nums)
        if fuel == 0:
            return 0
        if fuel == 1:
            return nums[start]
        if fuel >= 2:
            return max(
                nums[start] + recur(start + 1, fuel - 2),
                recur(start + 1, fuel - 1))

    return max(recur(i, starting_fuel) for i in range(len(nums)))


"""
SOLUTION WHEN THE BIRD CANNOT SKIP ANY TREES
"""

# TODO
