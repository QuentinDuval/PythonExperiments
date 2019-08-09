"""
https://leetcode.com/problems/maximum-sum-circular-subarray/
"""

import collections
from typing import List


class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        """
        General idea:
        - Keep on expanding the window while the prefix is > 0, else restart a new window
        - Remember the maximum sum encountered so far

        You can circle around the array, but never go past the starting point (length of slice):
        - you must reduce the window when it gets to big
        - PROBLEM: when you reduce the size of the array on the left, you need to also get rid of longest negative prefix

        !!!
        Complexity is O(N ** 2) unfortunately, cause the loop start_point_from might go all the way each time...
        => Need to find a better way to have the starting point
        !!!
        """

        '''
        n = len(nums)
        if n == 0:
            return 0

        max_sum = nums[0]
        lo = 0  # Inclusive start of range
        hi = 0  # Exclusive end of range
        curr_sum = 0 # Sum of nums[lo:hi]

        def start_point_from(lo, hi):
            # Try to find a new starting point: if any prefix is negative, skip it
            prefix_sum = 0
            curr_sum = 0
            new_lo = lo
            for i in range(lo, min(n, hi+1)):
                curr_sum += nums[i]
                if curr_sum <= 0:
                    new_lo = i + 1
                    prefix_sum += curr_sum
                    curr_sum = 0
            return new_lo, prefix_sum

        while hi < 2 * n and lo < n:
            i = hi if hi < n else hi - n
            curr_sum += nums[i]
            hi += 1

            if hi - lo > n:
                curr_sum -= nums[lo]
                lo += 1
                new_lo, prefix_sum = start_point_from(lo, hi)
                curr_sum -= prefix_sum
                lo = new_lo

            max_sum = max(max_sum, curr_sum)
            if curr_sum <= 0:
                curr_sum = 0
                lo = hi

        return max_sum
        '''

        N = len(nums)

        # Compute P[j] = sum(B[:j]) for the fixed array B = A+A
        prefix_sum = [0]
        for _ in range(2):
            for num in nums:
                prefix_sum.append(prefix_sum[-1] + num)

        # Want largest P[j] - P[i] with 1 <= j-i <= N
        # For each j, want smallest P[i] with i >= j-N
        max_sum = nums[0]

        # Stores the indices, such that always increasing P[i]
        deque = collections.deque([0])

        for j in range(1, len(prefix_sum)):

            # window is too big, reduce it by the negative prefix
            # - works because it is a mono-queue (monotonic queue)
            # - we get the next higher prefix_sum (negative prefix is removed)
            if deque[0] < j - N:
                deque.popleft()

            max_sum = max(max_sum, prefix_sum[j] - prefix_sum[deque[0]])

            # Extend the window with the number
            # - preserve the mono-queue (monotonic queue) invariant
            # - we want the smallest prefix_sum (so that we can remove it)
            # - it allows to skip the negative prefix (keep smallest because higher means you would decrease)
            while deque and prefix_sum[j] <= prefix_sum[deque[-1]]:
                deque.pop()
            deque.append(j)

        return max_sum

    # TODO - other solution is to split the problem in two:
    #   we know how to solve it for one simple range
    #   we know a buffer is two ranges split by one
    #   so we can try for every split (and try for the full array as well with single interval)
