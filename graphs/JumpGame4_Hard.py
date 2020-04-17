"""
https://leetcode.com/problems/jump-game-iv/

Given an array of integers arr, you are initially positioned at the first index of the array.

In one step you can jump from index i to index:
    i + 1 where: i + 1 < arr.length.
    i - 1 where: i - 1 >= 0.
    j where: arr[i] == arr[j] and i != j.

Return the minimum number of steps to reach the last index of the array.
Notice that you can not jump outside of the array at any time.
"""


from collections import *
from typing import *


class Solution:
    def minJumps(self, nums: List[int]) -> int:

        # Indexing of the positions (and prioritize the last ones)
        by_num = defaultdict(list)
        for pos, num in enumerate(nums):
            by_num[num].append(pos)
        for num, positions in by_num.items():
            positions.sort(reverse=True)

        # Get the neighbors (but erase the one discovered to make it faster)
        def neighbors_of(pos):
            if pos > 0:
                yield pos - 1
            if pos - 1 <= len(nums):
                yield pos + 1
            for neigh in by_num.get(nums[pos]):
                if neigh != pos:
                    yield neigh
            by_num[nums[pos]].clear() # Critical to pass the tests with lots of equal numbers

        # BFS to find the target node
        discovered = {0}
        to_visit = deque([(0, 0)])
        while to_visit:
            pos, dist = to_visit.popleft()
            if pos == len(nums) - 1:
                return dist
            for neigh in neighbors_of(pos):
                if neigh not in discovered:
                    to_visit.append((neigh, dist + 1))
                    discovered.add(neigh)
        return -1
