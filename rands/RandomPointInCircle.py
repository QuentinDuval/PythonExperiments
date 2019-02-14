"""
https://leetcode.com/problems/generate-random-point-in-a-circle
Given the radius and x-y positions of the center of a circle, write a function randPoint which generates a uniform random point in the circle.
"""


import math
import random
from typing import List


class Solution:
    def __init__(self, radius: float, x_center: float, y_center: float):
        self.radius = radius
        self.x_center = x_center
        self.y_center = y_center

    def randPoint(self) -> List[float]:
        """
        Generate the radius and the angle
        - angle is generated from uniform distribution
        - radius must not be uniform (otherwise more points to the center)
        Then transform to x, y
        """
        angle = random.uniform(0, 2 * math.pi)
        r2 = random.uniform(0, self.radius ** 2)
        x = math.sqrt(r2) * math.cos(angle)
        y = math.sqrt(r2) * math.sin(angle)
        return [self.x_center + x, self.y_center + y]