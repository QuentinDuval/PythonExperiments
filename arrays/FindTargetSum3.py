"""
Form with TREE NUMBERS:

https://practice.geeksforgeeks.org/problems/find-triplets-with-zero-sum/1

Given an array A of N elements.
Complete the function which returns true if triplets exists in array A whose sum is target else return false.
"""


from typing import List


def find_triple_sum(nums: List[int], target: int) -> bool:
    """
    One solution is to fall back on the solution of 'FindAllPairs':
    - one collection of sum of doubles in a hash map
    - one collection of simple sums from which we scan the list of doubles
    => O(N**2) space and time complexity

    Solution based on sorting:
    - Sort the collection
    - Move a first index 'i' from the lowest value to the top value
    - For each 'i', try to find two index j and k, such that i < j < k and the sum is the target
        - move 'j' from 'i' from low to high when the sum is lower
        - move 'k' from high to low when the sum is higher
    - This works and is equivalent to:
        - choose 'i' and 'j' such that i < k
        - binary search for 'k' such that i < j < k and the sum is target
    => O(N**2) time complexity but O(1) space complexity
    """

    nums.sort()
    if sum(nums[-3:0]) < target:
        return False

    # We could also start at index 'lo' where nums[lo] + nums[-2] + nums[-3] >= target (no solution otherwise)
    for lo in range(len(nums) - 2):
        if sum(nums[lo:lo+3]) > target:
            break

        # Could almost re-use the solution for 2, but for the sorting
        mid = lo + 1
        hi = len(nums) - 1
        while mid < hi:
            total = nums[lo] + nums[mid] + nums[hi]
            if total == target:
                return True
            elif total < target:
                mid += 1
            else:
                hi -= 1
    return False



