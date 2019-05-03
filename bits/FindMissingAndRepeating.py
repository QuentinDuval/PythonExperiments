"""
https://practice.geeksforgeeks.org/problems/find-missing-and-repeating/0

Given an unsorted array of size N of positive integers.
One number 'A' from set {1, 2, …N} is missing and one number 'B' occurs twice in array.
Find these two numbers.

Note:
- If you find multiple answers then print the Smallest number found.
- Also, expected solution is O(n) time and constant extra space.
"""


"""
How to approach it?
- We cannot sort because it is O(n log n)
- We cannot use a hash map because it would require extra space
=> We need to find two tricks to summarize the numbers, 2 equations to solve it

If we XOR the entire collection against the xor of 1 .. N:
=> you get the XOR of the two missing numbers (cause each appear an even number of time)

If we add the entire collection, we can compare the sum with N(N+1)/2 (the normal sum):
=> the difference is equal to "repeating - missing"

Then we can scan the numbers from 1 .. N, compute the XOR, and find if it is equal to the difference
"""


def missing_and_repeating(nums):
    n = len(nums)

    nums_xor = 0
    full_xor = 0
    nums_sum = 0
    for i, val in enumerate(nums):
        full_xor ^= (i + 1)
        nums_sum += val
        nums_xor ^= val

    diff_xor = full_xor ^ nums_xor
    diff_sum = nums_sum - n * (n + 1) / 2

    for a in range(1, n+1):
        b = diff_xor ^ a        # to get the other number
        if b - a == diff_sum:   # check if it matches the second equation
            if a not in nums:
                return b, a
    return None


# TODO - wrong complexity (too slow due to the "if a not in nums")


print(missing_and_repeating([1, 3, 3]))
# (3, 2)

print(missing_and_repeating([1, 14, 31, 8, 18, 33, 28, 2, 6, 16, 20, 3, 34, 17, 19, 21, 24, 25, 32, 11, 30, 13, 27, 7, 26, 29, 27, 15, 4, 12, 22, 5, 9, 10]))
# (27, 23)

