"""
https://leetcode.com/problems/erect-the-fence/

There are some trees, where each tree is represented by (x,y) coordinate in a two-dimensional garden.
Your job is to fence the entire garden using the minimum length of rope as it is expensive.
The garden is well fenced only if all the trees are enclosed.
Your task is to help find the coordinates of trees which are exactly located on the fence perimeter.
"""

from typing import List


class Solution:
    def outerTrees(self, points: List[List[int]]) -> List[List[int]]:
        """
        This is just the implementation of a Convex Hull
        But with a twist: the trees on the hull should be kept

        There are tons of algorithms for this:
        https://en.wikipedia.org/wiki/Convex_hull_algorithms
        """

        """
        Technique 1: Graham Scan (a variation is implemented below)
        https://en.wikipedia.org/wiki/Graham_scan

        - Select a point P to be the leftmost point (and highest if equality)
        - Sort the points by angles with P, and consider them in increasing angle order [-pi, pi]
        - When considering a new point, check whether:
          - it turns left (keep the previous)
          - it turns right (drop the previous)
          - if it goes straight (keep the previous - specificity of the problem)
        - The check should be done recursively (keep a stack)

        To order the points, the angles are not needed:
        - Any monotonic function of [-pi, pi] is enough (the angle is not needed).
        - In particular, we can simply order by slope

        To check for the turn, we do not really need the angle either:
        - We can use any function that is monotonic in [-pi/2, pi/2] like 'sin'
        - We can use the cross product and check if it is positive or negative
          (Since it is equal to norm(x) * norm(y) * sin(angle between x and y) * z)

        !!! There might be colinar vectors !!!
        - we want the points the closest to the source when going away
        - we want the points the furthest to the source when going back
        => we order by slope then highest 'y' coordinate first
        => but it does not work for equal 'y' so we do the Graham scan in BOTH direction...
        """
        if len(points) <= 3:
            return points

        points = [tuple(p) for p in points]
        ref = self.reference_point(points)
        self.sort_by_slopes(ref, points, 1)
        frontier = set(self.graham_scan(ref, points, self.is_turn_right))
        self.sort_by_slopes(ref, points, -1)
        frontier |= set(self.graham_scan(ref, points, self.is_turn_left))
        return list(frontier)

    def graham_scan(self, ref, points, keep_turn):
        stack = [ref, points[0]]
        for p in points[1:]:
            while len(stack) >= 2 and keep_turn(stack[-2], stack[-1], p):
                stack.pop()
            stack.append(p)
        return stack

    def reference_point(self, points):
        ref = min(points)
        points.pop(points.index(ref))
        return ref

    def sort_by_slopes(self, ref, points, order):
        points.sort(key=lambda p: (order * self.slope(ref, p), -p[1]))

    def slope(self, ref, p):
        if ref[0] == p[0]:
            return float('inf')
        else:
            return (p[1] - ref[1]) / (p[0] - ref[0])

    def is_turn_right(self, p1, p2, p3):
        return self.cross_product(self.direction(p1, p2), self.direction(p1, p3)) < 0

    def is_turn_left(self, p1, p2, p3):
        return self.cross_product(self.direction(p1, p2), self.direction(p1, p3)) > 0

    def direction(self, x, y):
        return [y[0] - x[0], y[1] - x[1]]

    def cross_product(self, x, y):
        return x[0] * y[1] - x[1] * y[0]


"""
TO DEBUG
"""


'''
import matplotlib.pyplot as plot


def show_diff(points, expected):
    points = [(x, y) for x, y in points]
    expected = [(x, y) for x, y in expected]
    missing = set(expected) - set(points)

    plot.scatter([p[0] for p in points], [p[1] for p in points], c='b', marker='.')
    plot.scatter([p[0] for p in missing], [p[1] for p in missing], c='r', marker='+')
    plot.show()


show_diff(
    points=[[0,0],[8,0],[9,6],[7,9],[1,9]],
    expected=[[9,6],[7,9],[1,0],[1,9],[2,0],[8,0],[0,0],[3,0]]
)
'''
