import abc
from collections import *
from dataclasses import *
import enum
import numpy as np
import time
from typing import *
import sys
import math


"""
------------------------------------------------------------------------------------------------------------------------
UTILS
------------------------------------------------------------------------------------------------------------------------
"""


def debug(*args):
    print(*args, file=sys.stderr)


T = TypeVar('T')


class Chronometer:
    def __init__(self):
        self.start_time = 0

    def start(self):
        self.start_time = time.time_ns()

    def spent(self):
        current = time.time_ns()
        return self._to_ms(current - self.start_time)

    def _to_ms(self, delay):
        return delay / 1_000_000


"""
------------------------------------------------------------------------------------------------------------------------
BASIC GEOMETRY
------------------------------------------------------------------------------------------------------------------------
"""


Angle = float
Duration = float
Mass = float
Vector = np.ndarray


def vector(x: int, y: int) -> Vector:
    return np.array([x, y])


def norm2(v) -> float:
    return np.dot(v, v)


def norm(v) -> float:
    return math.sqrt(norm2(v))


def distance2(v1: Vector, v2: Vector) -> float:
    v = v1 - v2
    return np.dot(v, v)


def distance(v1: Vector, v2: Vector) -> float:
    return math.sqrt(distance2(v1, v2))


def get_angle(v: Vector) -> Angle:
    # Get angle from a vector (x, y) in radian
    x, y = v
    if x > 0:
        return np.arctan(y / x)
    if x < 0:
        return math.pi - np.arctan(- y / x)
    return math.pi / 2 if y >= 0 else -math.pi / 2


def mod_angle(angle: Angle) -> Angle:
    if angle > 2 * math.pi:
        return angle - 2 * math.pi
    if angle < 0:
        return angle + 2 * math.pi
    return angle


"""
------------------------------------------------------------------------------------------------------------------------
GAME MECANICS
------------------------------------------------------------------------------------------------------------------------
"""


@dataclass(frozen=False)
class Entities:
    positions: np.ndarray
    speeds: np.ndarray
    radius: np.ndarray
    masses: np.ndarray


def normal_of(p1: Vector, p2: Vector) -> Vector:
    n = np.array([p1[1] - p2[1], p2[0] - p1[0]])
    return n / norm(n)


def find_collision(entities: Entities, i1: int, i2: int, dt: float) -> float:

    # Change referential to i1 => subtract speed of i1 to i2
    # The goal will be to check if p1 intersects p2-p3
    p1 = entities.positions[i1]
    p2 = entities.positions[i2]
    speed = entities.speeds[i2] - entities.speeds[i1]
    p3 = p2 + speed * dt

    # Quick collision check: check the distances
    d13 = distance2(p3, p1)
    d23 = distance2(p3, p2)
    d12 = distance2(p1, p2)
    if max(d12, d13) > d23:
        return float('inf')

    # Check the distance of p1 to p2-p3
    n = normal_of(p2, p3)
    dist_to_segment = abs(np.dot(n, p1 - p2))
    sum_radius = entities.radius[i1] + entities.radius[i2]
    if dist_to_segment > sum_radius:
        return float('inf')

    # Find the point of intersection (a bit of trigonometry and pythagoras involved)
    distance_to_normal = np.dot(p1 - p2, p3 - p2) / math.sqrt(d23)
    distance_to_intersection: float = distance_to_normal - math.sqrt(sum_radius ** 2 - dist_to_segment ** 2)
    return distance_to_intersection / norm(speed)


def simulate_collisions(entities: Entities, dt: float = 1.0) -> Entities:
    pass
