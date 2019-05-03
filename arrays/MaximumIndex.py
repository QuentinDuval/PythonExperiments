"""
https://practice.geeksforgeeks.org/problems/maximum-index/0

Given an array A[] of N positive integers.
The task is to find the maximum of j - i subjected to the constraint of A[i] <= A[j].
"""


"""
How to approach it?

The brute force approach is to try for every i < j, but:
- It is useless to try at every i: we try only at a successive index 'i' if its value is LOWER than all BEFORE
- It is useless to try at every j: we try only at a preceding index 'j' if its value is HIGHER than all AFTER
=> Note there is a kind of symmetry.

Still, even if we only try these combinations, the complexity is O(N**2):
- Example of the collection [5, 10, 4, 9, 3, 8, 2, 7, 1, 6]

SOLUTION:
- We keep an ordered map of values collected from right to left (we only add values higher than all previous) to index
- We scan from left to right, and try each value (search lower_bound) and compute the difference of index

EXAMPLE:
[34, 8, 10, 3, 2, 80, 30, 33, 1]
- map contains [1 => 8, 33 => 7, 80 => 5] (this is just a vector & we can binary search in it)
- then we look for 34 at index 0 => we get 80 at index 5 => 5
- then we look for 8 at index 1 => we get 33 at index 7 => 6
- and so on
"""


def maximum_index(nums):
    pass


# TODO


print(maximum_index([34, 8, 10, 3, 2, 80, 30, 33, 1]))  # Expect 6

