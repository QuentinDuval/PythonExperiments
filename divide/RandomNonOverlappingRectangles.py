"""
https://leetcode.com/problems/random-point-in-non-overlapping-rectangles
"""


from typing import List
import random


class Solution:
    """
    Since the rectangles are not overlapping:
    - We can just ponder with their size
    - And pick a number between 0 and the sum of their size
    - Use a binary search then to find the right rectangle
    - Then pick a number randomly inside the rectangle
    """

    def __init__(self, rects: List[List[int]]):
        self.sizes = []
        self.rects = rects
        self.total_size = 0
        for x1, y1, x2, y2 in rects:
            self.total_size += (y2 - y1 + 1) * (x2 - x1 + 1)
            self.sizes.append(self.total_size)

    def pick(self) -> List[int]:
        x1, y1, x2, y2 = self.random_rectangle()
        return [random.randint(x1, x2), random.randint(y1, y2)]

    def random_rectangle(self):
        lo = 0
        hi = len(self.sizes) - 1
        val = random.randint(0, self.total_size - 1)
        while lo <= hi:
            mid = (hi + lo) // 2
            if val < self.sizes[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        return self.rects[lo]
