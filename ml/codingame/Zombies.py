import math
import sys
import time
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

    @staticmethod
    def _to_ms(delay):
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
    return float(np.dot(v, v))


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

MAP_WIDTH = 16000
MAP_HEIGHT = 9000


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
        entity_id, x, y = [int(j) for j in input().split()]
        return cls(id=entity_id, position=np.array([x, y]))


@dataclass()
class Zombie:
    id: EntityId
    position: Vector
    next_position: Vector

    @classmethod
    def read(cls):
        entity_id, x, y, x_next, y_next = [int(j) for j in input().split()]
        return cls(id=entity_id, position=np.array([x, y]), next_position=np.array([x_next, y_next]))


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
        player_pos = game_state.player

        # Find the human that you can really save
        savable: List[Human] = []
        for h in game_state.humans:
            closest_zombie = closest(game_state.zombies, h.position)
            closest_distance = distance(closest_zombie.position, h.position)
            if can_save_human(closest_distance, distance(game_state.player, h.position)):
                savable.append(h)

        # The human to protect and its closest zombie
        closest_human = closest(savable, game_state.player) # TODO - might crash, but should not if you do it right
        closest_zombie = closest(game_state.zombies, closest_human.position)

        # If the zombie is closer than the human to defend
        if distance2(player_pos, closest_zombie.position) < distance2(player_pos, closest_human.position):

            # If the player is in between, the human is safe for now, try to collect max zombies
            my_zombies = self.zombies_on_player(game_state)
            if my_zombies and distance2(player_pos, closest_human.position) < distance2(closest_human.position, closest_zombie.position):
                target_position = self.find_attaction_point(my_zombies)

            # Go directly to intercept the zombie
            else:
                target_position = closest_zombie.next_position

        # Otherwise, go to protect the human by intercepting the zombie
        else:
            target_position = (closest_human.position + closest_zombie.position) / 2
        debug("Time spent:", self.chrono.spent(), "ms")
        return target_position

    def zombies_on_player(self, game_state: GameState) -> List[Zombie]:
        zombies = []
        for z in game_state.zombies:
            h = closest(game_state.humans, z.position)
            if distance2(h.position, z.position) > distance2(game_state.player, z.position):
                zombies.append(z)
        return zombies

    def find_attaction_point(self, zombies: List[Zombie]) -> Vector:
        best_point = None
        best_value = float('inf')
        xs = np.random.uniform(0, MAP_WIDTH, size=500)
        ys = np.random.uniform(0, MAP_HEIGHT, size=500)
        for x, y in zip(xs, ys):
            point = np.array([x, y])
            distances = np.array([distance(z.position, point) for z in zombies])
            mean = np.mean(distances)
            variance = np.mean(distances ** 2) - np.mean(distances) ** 2
            if mean + variance < best_value:
                best_value = variance + mean
                best_point = point
        return best_point


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
        print(int(target[0]), int(target[1]), "I love zombies")


if __name__ == '__main__':
    game_loop()
