"""
https://leetcode.com/problems/count-number-of-nice-subarrays

Given an array of integers nums and an integer k. A subarray is called nice if there are k odd numbers on it.

Return the number of nice sub-arrays.
"""


from typing import List


class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        """
        IDEA:
        We must find exactly k=3 odd numbers:
        - note that just counting - k + 1 does not work cause of even values in between
        - so a window would work: NAIVE DOES NOT WORK - cause misses the "hi" movement

        Complexity is O(N)
        """
        combi = 0

        lo = 0
        hi = 0
        count = 0
        while hi < len(nums):
            if nums[hi] & 1:
                count += 1
            if count < k:
                hi += 1
            else:
                factor = 1
                hi += 1
                while hi < len(nums) and nums[hi] & 1 == 0:
                    hi += 1
                    factor += 1
                while count >= k:
                    combi += factor
                    if nums[lo] & 1:
                        count -= 1
                    lo += 1
        return combi

        """
        Idea with mono-queue?
        """
