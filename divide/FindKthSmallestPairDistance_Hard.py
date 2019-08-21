"""
https://leetcode.com/problems/find-k-th-smallest-pair-distance/

Given an integer array, return the k-th smallest distance among all the pairs.
The distance of a pair (A, B) is defined as the absolute difference between A and B.
"""


from typing import List


class Solution:
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        """
        Brute force is to generate all pairs, then sort them, then take the Kth pair
        and return its distance.
        => Complexity is O(N^2 * log N)

        We could also use a heap, and heapify the N * N pairs, and then get K of them
        => Complexity is O(N^2 + K log N)

        A last approach consists in "guessing" what is the distance, and binary search
        for that. To check if it is the correct distance:
        - sort the nums
        - search for each num the number of elements below num + dist
        - if the total number of element is lower than k, increase the distance, else decrease it
        => Complexity is O(N log N log MaxDist)

        Beats 5%, 672 ms

        Because we can do better:
        - The binary searches will only increase (for each num).
        - So we could in fact do a window approach for O(N log MaxDist)
        """
        nums.sort()

        lo_dist = 0  # min distance
        hi_dist = nums[-1] - nums[0]  # max possible distance
        while lo_dist <= hi_dist:
            mid_dist = lo_dist + (hi_dist - lo_dist) // 2
            lower = self.count_lower(nums, mid_dist)
            if lower < k:  # count itself as lower number
                lo_dist = mid_dist + 1
            else:
                hi_dist = mid_dist - 1

        # Take the smallest distance below lo_dist (it must be a real one)
        # But it will be a real one (cause we keep on dividing, until partition point)
        return lo_dist

    def count_lower(self, nums, distance):
        count = 0
        for i, num in enumerate(nums):
            next_bigger = self.upper_bound(nums, num + distance)
            count += next_bigger - i - 1
        return count

    def upper_bound(self, nums, val):
        lo = 0
        hi = len(nums) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if nums[mid] <= val:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo
