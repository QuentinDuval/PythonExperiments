"""
https://practice.geeksforgeeks.org/problems/subarray-with-given-sum/0

Given an unsorted array A of size N of non-negative integers, find a continuous sub-array which adds to a given number.
"""


from typing import List


def subarray_with_given_sum(nums: List[int], target: int):
    """
    Brute force algorithm would consist in trying all i < j: O(N^2)

    But if i < j is lower than target:
    - it is useless to try inside
    - we should only try to extend the window

    If we start the window i < j at index i = 0 and j = 0:
    - Just extend the window to the right j => j + 1 if sum is lower than target
    - Just restrict the window to the right i => i + 1 if sum is higher than target

    We cannot miss a case:
    - For a given 'i', no need to test more to the right if sum > target
    - For a given 'j', no need to test more to the left if sum > target
    """
    start = 0
    cum_sum = 0
    for end in range(len(nums)):
        cum_sum += nums[end]
        while cum_sum > target and start < end:
            cum_sum -= nums[start]
            start += 1

        # Must absolutely follow the 'while' in order not to miss a case
        if cum_sum == target:
            return start, end

    return (-1,)


def subarray_with_given_sum_2(n: int, target: int, nums: List[int]):
    """
    OTHER TECHNIQUE: binary searching
    - Compute the partial cumulative sums
    - For each i, binary search for the target + cum_sum[i-1]
    => Complexity is O(n log n)
    """
    if n <= 0:
        return 0, 0

    cum_sums = [nums[0]]
    for num in nums[1:]:
        cum_sums.append(cum_sums[-1] + num)

    def binarySearch(val):
        lo = 0
        hi = len(cum_sums) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if cum_sums[mid] == val:
                return mid
            elif cum_sums[mid] < val:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    for i in range(n):
        shift = cum_sums[i-1] if i > 0 else 0
        j = binarySearch(target + shift)
        if j is not None:
            return i, j
    return None

