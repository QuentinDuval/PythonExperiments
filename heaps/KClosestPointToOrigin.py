"""
https://leetcode.com/problems/k-closest-points-to-origin/
"""

import heapq
from typing import List


class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        if k <= 0:
            return []

        if k == 1:
            return [min(points, key=lambda p: self.distance(p[0], p[1]))]

        if k >= len(points) // 4:
            points.sort(key=lambda p: self.distance(p[0], p[1]))
            return points[:k]

        queue = [(self.distance(x, y), x, y) for x, y in points]
        heapq.heapify(queue)
        closest = []
        for _ in range(k):
            d, x, y = heapq.heappop(queue)
            closest.append([x, y])
        return closest

    def distance(self, x, y):
        return x ** 2 + y ** 2
