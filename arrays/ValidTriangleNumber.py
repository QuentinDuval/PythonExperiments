"""
https://leetcode.com/problems/valid-triangle-number/

Given an array consists of non-negative integers, your task is to count the number of triplets chosen
from the array that can make triangles if we take them as side lengths of a triangle.
"""


import bisect
from typing import List


class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        """
        A triangle is valid if the length of the biggest side is strictly lower
        than the sum of the length of the two smallest sides.

        Sort the numbers:
        - pick the smallest side
        - pick the second smallest side
        - select all bigger side with sum < sum(other sides)

        Since we do not need to find the triangles but count them, we can just
        do a binary search to find the last valid triangle biggest side
        """

        # TODO - can do better: no binary search - just advance the bigger side (it necessarily grows) => O(N ** 2)

        n = len(nums)
        if n < 3:
            return 0

        def lower_bound(val: int, lo: int):
            hi = n - 1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if nums[mid] < val:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return lo

        triangle_count = 0
        nums.sort()
        for i in range(n - 2):
            if nums[i] <= 0:
                continue

            for j in range(i + 1, n - 1):
                side_sum = nums[i] + nums[j]
                k = bisect.bisect_left(nums, side_sum)
                # k = lower_bound(side_sum, lo=j)
                if k > j:
                    triangle_count += (k - 1 - j)
        return triangle_count
