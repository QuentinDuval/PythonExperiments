"""
https://leetcode.com/problems/valid-square/

Given the coordinates of four points in 2D space, return whether the four points could construct a square.

The coordinate (x,y) of a point is represented by an integer array with two integers.
"""


from typing import List


class Solution:
    def validSquare(self, p1: List[int], p2: List[int], p3: List[int], p4: List[int]) -> bool:
        """
        Property of a square:
        - all sides of the same length
        - all angles are pi/2
        We could try all 4! permutations... but you can just fix 1 point and try 3! permutations.

        You might think there is some subtelty, but there is not. You have to try everything.
        Example of  one FALSE algorithm:
        - take one point as reference p1
        - then compute the distance of other points p2, p3, p4 to p1
        - one of them should be at distance sqrt(2) bigger than the others (which should be equal)
        If so, you have a square, else, you do not.
        => does not work (ex of one point that is at the good distance but not on the square)
        """
        return any(
            self.is_rectangle(combi)
            for combi in
            [[p1, p2, p3, p4],
             [p1, p2, p4, p3],
             [p1, p3, p2, p4],
             [p1, p3, p4, p2],
             [p1, p4, p2, p3],
             [p1, p4, p3, p2]])

    def is_rectangle(self, points: List[List[int]]):
        distance = self.sq_dist(points[-1], points[0])
        if distance == 0:
            return False

        vect = self.diff(points[-1], points[0])
        for i in range(len(points) - 1):
            if self.sq_dist(points[i], points[i + 1]) != distance:
                return False
            vect = self.rotate(vect)
            if vect != self.diff(points[i], points[i + 1]):
                return False
        return True

    def sq_dist(self, p1: List[int], p2: List[int]):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def diff(self, origin: List[int], destination: List[int]) -> List[int]:
        return [destination[0] - origin[0], destination[1] - origin[1]]

    def rotate(self, v: List[int]) -> List[int]:
        x, y = v
        return [-y, x]

