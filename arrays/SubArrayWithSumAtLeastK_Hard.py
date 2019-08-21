"""
https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/

Return the length of the shortest, non-empty, contiguous subarray of A with sum at least K.

If there is no non-empty subarray with sum at least K, return -1.
"""


from collections import deque
from typing import List


class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        """
        The two pointer approach (grow when below, shrink when above) does
        not work here, because of the negative numbers.

        But a similar approach does, using MonoQueues (monotonic queues).

        Take the example of:
        [2, 1, 14, -8, 14, 3] and K = 16

        Take a look at the prefix sums:
        [0, 2, 3, 17, 9, 23, 26]

        It should be pretty clear that if we want to advance our start pointer,
        starting from 9 is better than 17:
        - it will lead to a shorter window
        - it will lead to a bigger sum

        So the algorithm is pretty simple:
        - add new elements to the window when we are below the sum
        - when we are above, try to shrink it:
            - go to the next higher element on prefix sum to the right
            - it will have lower sum, but shorter window (might be good)

        To make this efficient, we use a mono-queue:
        - when we add an element, we pop lower elements at the end of the queue
        - we keep the index of the elements in the queue (to identify window)

        Complexity is O(N) time, and O(N) space
        """

        # TODO - we could get rid of prefix sums (but requires storing value + index in monoqueue)

        prefix_sums = [0]
        for num in nums:
            prefix_sums.append(prefix_sums[-1] + num)

        shortest = float('inf')
        monoqueue = deque([0])  # index in the prefix sum
        for end in range(1, len(prefix_sums)):

            # Shorten the window if sum is bigger
            while prefix_sums[end] - prefix_sums[monoqueue[0]] >= k:
                shortest = min(shortest, end - monoqueue[0])
                if len(monoqueue) > 1:  # Array must be non-null (need a start)
                    monoqueue.popleft()
                else:
                    break

            # Maintain the invariant of monotonic queue
            while monoqueue and prefix_sums[monoqueue[-1]] >= prefix_sums[end]:
                monoqueue.pop()
            monoqueue.append(end)

        return shortest if shortest != float('inf') else -1


