"""
https://leetcode.com/problems/contains-duplicate-iii/

Given an array of integers, find out whether there are two distinct indices i and j in the array such that the
absolute difference between nums[i] and nums[j] is at most t and the absolute difference between i and j is at most k.
"""


from typing import List


class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        """
        The standard solution is to use a SORTED MAP:
        - initialize the sorted map with the first 'k' elements
        - add elements in the sorted map at index 'i'
        - remove elements in the sorted map at index 'i-k'
        - every time you add an element 'n', check if [n-t, n+t] is empty or not
          this can be done in O(log N) time in a sorted map
        => total complexity is O(N log N)

        The bucket sort solution is clever and runs in O(N):
        - partition the range of values in buckets of width 't' (add at key 'num / t')
        - if two elements fall in the same bucket, it is a match
        - we also have to check elements in the two nearby buckets
        Because we stop at first collision, there is no need to worry about multiple
        elements by buckets.
        """

        if t < 0:
            return False

        buckets = {}
        width = t + 1

        for i, num in enumerate(nums):
            start_window = i - k
            if start_window > 0:
                del buckets[nums[start_window - 1] // width]

            bucket = num // width
            if bucket in buckets:
                return True

            left = buckets.get(bucket - 1)
            if left is not None and num - left <= t:
                return True

            right = buckets.get(bucket + 1)
            if right is not None and right - num <= t:
                return True

            buckets[bucket] = num

        return False


