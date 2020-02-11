import copy
import enum
import heapq
import math
import sys
import time
from collections import *
from dataclasses import *
from typing import *

import numpy as np


"""
------------------------------------------------------------------------------------------------------------------------
UTILS
------------------------------------------------------------------------------------------------------------------------
"""


def debug(*args):
    print(*args, file=sys.stderr)


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
GEOMETRY & VECTOR CALCULUS
------------------------------------------------------------------------------------------------------------------------
"""


Angle = float
Vector = np.ndarray


def get_angle(v: Vector) -> Angle:
    # Get angle from a vector (x, y) in radian
    x, y = v
    if x > 0:
        return np.arctan(y / x)
    if x < 0:
        return math.pi - np.arctan(- y / x)
    return math.pi / 2 if y >= 0 else -math.pi / 2


def norm2(v) -> float:
    return np.dot(v, v)


def norm(v) -> float:
    return math.sqrt(norm2(v))


def distance2(from_, to_) -> float:
    return norm2(to_ - from_)


def distance(from_, to_) -> float:
    return norm(to_ - from_)


def mod_angle(angle: Angle) -> Angle:
    if angle > 2 * math.pi:
        return angle - 2 * math.pi
    if angle < 0:
        return angle + 2 * math.pi
    return angle


"""
------------------------------------------------------------------------------------------------------------------------
GAME CONSTANTS
------------------------------------------------------------------------------------------------------------------------
"""


ASH_SPEED = 1000
ZOMBIE_SPEED = 400


"""
------------------------------------------------------------------------------------------------------------------------
GAME ENTITIES
------------------------------------------------------------------------------------------------------------------------
"""


T = TypeVar('T')


EntityId = int


@dataclass()
class Human:
    id: EntityId
    position: Vector

    @classmethod
    def read(cls):
        id, x, y = [int(j) for j in input().split()]
        return cls(id=id, position=np.array([x, y]))


@dataclass()
class Zombie:
    id: EntityId
    position: Vector
    next_position: Vector

    @classmethod
    def read(cls):
        id, x, y, x_next, y_next = [int(j) for j in input().split()]
        return cls(id=id, position=np.array([x, y]), next_position=np.array([x_next, y_next]))


@dataclass()
class GameState:
    player: Vector
    humans: List[Human]
    zombies: List[Zombie]

    @classmethod
    def read(cls):
        x, y = [int(i) for i in input().split()]
        return cls(
            player=np.array([x, y]),
            humans=[Human.read() for _ in range(int(input()))],
            zombies=[Zombie.read() for _ in range(int(input()))]
        )


def closest(entities: List[T], position: Vector) -> T:
    return min(entities, key=lambda e: distance2(e.position, position))


"""
------------------------------------------------------------------------------------------------------------------------
AGENT
------------------------------------------------------------------------------------------------------------------------
"""


def can_save_human(zombie_dist, player_dist):
    return (zombie_dist / ZOMBIE_SPEED) > (player_dist / ASH_SPEED - 2)  # Minus the number of turns to shoot


class Agent:
    def __init__(self):
        self.chrono = Chronometer()

    def get_action(self, game_state: GameState) -> Vector:
        self.chrono.start()
        savable: List[Human] = []
        for h in game_state.humans:
            closest_zombie = closest(game_state.zombies, h.position)
            closest_distance = distance(closest_zombie.position, h.position)
            if can_save_human(closest_distance, distance(game_state.player, h.position)):
                savable.append(h)

        closest_human = closest(savable, game_state.player)
        debug("Time spent:", self.chrono.spent(), "ms")
        return closest_human.position


"""
------------------------------------------------------------------------------------------------------------------------
GAME LOOP
------------------------------------------------------------------------------------------------------------------------
"""


def game_loop():
    agent = Agent()
    while True:
        game_state = GameState.read()
        target = agent.get_action(game_state)
        print(target[0], target[1], "I love zombies")


if __name__ == '__main__':
    game_loop()
