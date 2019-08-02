"""
https://leetcode.com/problems/max-points-on-a-line

Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.
"""

from typing import List


class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        multiplicity = {}
        for p in points:
            multiplicity[(p[0], p[1])] = multiplicity.get((p[0], p[1]), 0) + 1

        points = list(multiplicity.keys())
        n = len(points)
        if n <= 2:
            return sum(multiplicity.values())

        lines = {}
        max_count = 0
        for i1, p1 in enumerate(points[:-1]):
            from_i1 = set()
            for p2 in points[i1 + 1:]:
                line = get_line(p1, p2)
                if line in from_i1:  # Because the previous i2 found from i1 will also find this i2 (kind of a HACK)
                    continue

                count = lines.get(line, multiplicity[p1]) + multiplicity[p2]
                max_count = max(max_count, count)
                lines[line] = count
                from_i1.add(line)
        return max_count


def get_line(p1, p2):
    # Deducing y = a x + b (but for vertical lines)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0:
        return (p1[0], 0)
    if dy == 0:
        return (0, p1[1])
    else:
        a = Ratio(dy, dx)
        b = Ratio(p1[1]) - a * Ratio(p1[0])
        return (a, b)


def gcd(a, b):
    a, b = max(a, b), min(a, b)
    while a % b != 0:
        a, b = b, a % b
    return b


def lcm(a, b):
    return a * b // gcd(a, b)


class Ratio:
    def __init__(self, a, b=1):
        if b < 0:
            b = -1 * b
            a = -1 * a
        if a == 0:
            self.num = 0
            self.denum = 1
        else:
            g = gcd(a, b)
            self.num = a // g
            self.denum = b // g

    def __hash__(self):
        return hash((self.num, self.denum))

    def __eq__(self, other):
        return self.num == other.num and self.denum == other.denum

    def __repr__(self):
        return str(self.num) + "/" + str(self.denum)

    def __mul__(self, other):
        return Ratio(self.num * other.num, self.denum * other.denum)

    def __add__(self, other):
        l = lcm(self.denum, other.denum)
        return Ratio(self.num * l // self.denum + other.num * l // other.denum, l)

    def __sub__(self, other):
        l = lcm(self.denum, other.denum)
        return Ratio(self.num * l // self.denum - other.num * l // other.denum, l)


