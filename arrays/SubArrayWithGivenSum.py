"""
https://practice.geeksforgeeks.org/problems/subarray-with-given-sum/0

Given an unsorted array A of size N of non-negative integers, find a continuous sub-array which adds to a given number.
"""


from typing import List


def subarrayWithSum(n: int, target: int, nums: List[int]):
    """
    Brute force way:
    - Try all i < j and compute the sum => see if it fits the target
    => Complexity is O(n^3)

    Dynamic programming:
    - Try all i < j and compute the sum => see if it fits the target
    - Precompute the partial sums so that the sum between i < j is O(1)
    => Complexity is O(n^2)

    Binary searching:
    - Precompute the partial sums
    - For each i, binary search for the target + sum up to i-1
    => Complexity is O(n log n)

    One pass algorithm in O(n):
    - Increase j if the sum between i < j is inferior to target
    - Increase i if the sum between i < j is superior to target
    Why does it work?
    - For a fixed 'i', there is no point in going after 'j' if superior.
    - So you just move to next 'i'... but can we might miss element?
    - No! The sub-array was nessarily not big enough (we went for 'j')
    """
    if n <= 0:
        return 0, 0

    '''
    # Binary search based
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
    '''

    i = j = 0
    partial_sum = nums[0]
    while j < n:
        if partial_sum == target:
            return i, j

        if partial_sum < target:
            j += 1
            if j < n:
                partial_sum += nums[j]
        else:
            partial_sum -= nums[i]
            i += 1
    return None


'''
# Test client for Geeks for Geeks
test_nb = int(input())
for _ in range(test_nb):
    line1 = input().strip()
    line2 = input().strip()
    n, target = [int(x) for x in line1.split(" ")]
    nums = [int(x) for x in line2.split(" ")]
    result = subarrayWithSum(n, target, nums)
    if result is not None:
        start, end = result
        print(start + 1, end + 1)
    else:
        print(-1)
'''
