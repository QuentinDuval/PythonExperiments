"""
https://leetcode.com/problems/find-k-closest-elements/

Given a sorted array, two integers k and x, find the k closest elements to x in the array.
The result should also be sorted in ascending order.
If there is a tie, the smaller elements are always preferred.
"""


from collections import deque
import heapq
from typing import List


class Solution:
    def findClosestElements_heap(self, nums: List[int], k: int, x: int) -> List[int]:
        """
        Binary search the position of the x element and use a heap to get the
        closest elements around it.

        Complexity is O(log N + K log K)
        Beats 10%, 440ms
        """

        # Easy cases
        if not nums:
            return nums
        if x <= nums[0]:
            return nums[:k]
        if x >= nums[-1]:
            return nums[-k:]

        # Binary searching 'x' then using a heap
        # Another solution is to take the number in order from x to left and x to right and merge
        lo = self.lower_bound(nums, x)
        heap = []
        for i in range(max(0, lo - k), min(len(nums), lo + k + 1)):
            heapq.heappush(heap, (abs(x - nums[i]), nums[i]))

        selected = []
        while len(selected) < k:
            _, val = heapq.heappop(heap)
            selected.append(val)
        selected.sort()
        return selected

    def findClosestElements_merge(self, nums: List[int], k: int, x: int) -> List[int]:
        """
        Now, instead of a heap, we could use a simple merge sort from the point x
        we have found.

        Complexity is O(log N + K)
        Beats 57%, 360ms
        """

        # Easy cases
        if not nums:
            return nums
        if x <= nums[0]:
            return nums[:k]
        if x >= nums[-1]:
            return nums[-k:]

        # Merge sort from the point 'mid'
        hi = self.lower_bound(nums, x)

        selected = deque()
        lo = hi - 1
        while len(selected) < k:
            if lo < 0:
                selected.append(nums[hi])
                hi += 1
            elif hi == len(nums) or x - nums[lo] <= nums[hi] - x:
                selected.appendleft(nums[lo])
                lo -= 1
            else:
                selected.append(nums[hi])
                hi += 1
        return list(selected)

    def findClosestElements_bad_bsearch(self, nums: List[int], k: int, x: int) -> List[int]:
        """
        Now, instead of doing a merge sort, we could do a binary search on the
        delta from 'x' and to 'x', and binary search 'x+delta' and 'x-delta'.

        But it does not work (because we have equal numbers...)
        """

        # Easy cases
        if not nums:
            return nums
        if x <= nums[0]:
            return nums[:k]
        if x >= nums[-1]:
            return nums[-k:]

        # Binary search in binary search (oh yeah)
        delta_lo = 0
        delta_hi = 10000
        while delta_lo <= delta_hi:
            delta = delta_lo + (delta_hi - delta_lo) // 2
            lo = self.lower_bound(nums, x - delta)
            hi = self.lower_bound(nums, x + delta - 1)
            if hi - lo == k:
                return nums[lo:hi]
            elif hi - lo < k:
                delta_lo = delta + 1
            else:
                delta_hi = delta - 1

    def findClosestElements(self, nums: List[int], k: int, x: int) -> List[int]:
        """
        Instead, we can binary search the start of the window directly.

        Complexity is O(log N + K), same as before.
        Beats 99.67% (324 ms)
        """

        # Easy cases
        if not nums:
            return nums
        if x <= nums[0]:
            return nums[:k]
        if x >= nums[-1]:
            return nums[-k:]

        # Search the starting point of the window
        lo = 0
        hi = len(nums) - k - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            # If delta is too big on left side, we should try left
            if x - nums[mid] > nums[mid + k] - x:
                lo = mid + 1
            else:
                hi = mid - 1
        return nums[lo:lo + k]

    def lower_bound(self, nums, val):
        lo = 0
        hi = len(nums) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if nums[mid] < val:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo
