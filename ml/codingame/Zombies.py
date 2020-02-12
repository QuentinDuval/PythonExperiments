import itertools
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
SHOOTING_RADIUS = 2000

MAP_WIDTH = 16000
MAP_HEIGHT = 9000

MAX_RESPONSE_TIME = 100


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
        return cls(id=entity_id,
                   position=np.array([x, y], dtype=np.float64))

    def __eq__(self, other):
        return self.id == other.id and np.array_equal(self.position, other.position)

    def copy(self, **kwargs):
        return replace(self, **kwargs)


@dataclass()
class Zombie:
    id: EntityId
    position: Vector
    next_position: Vector

    @classmethod
    def read(cls):
        entity_id, x, y, x_next, y_next = [int(j) for j in input().split()]
        return cls(id=entity_id,
                   position=np.array([x, y], dtype=np.float64),
                   next_position=np.array([x_next, y_next], dtype=np.float64))

    def __eq__(self, other):
        return self.id == other.id and np.array_equal(self.position, other.position) and np.array_equal(self.next_position, other.next_position)

    def copy(self, **kwargs):
        return replace(self, **kwargs)


@dataclass()
class GameState:
    player: Vector
    humans: List[Human]
    zombies: List[Zombie]

    @classmethod
    def read(cls):
        x, y = [int(i) for i in input().split()]
        return cls(
            player=np.array([x, y], dtype=np.float64),
            humans=[Human.read() for _ in range(int(input()))],
            zombies=[Zombie.read() for _ in range(int(input()))])

    def clone(self):
        return GameState(
            player=self.player.copy(),
            humans=[h.copy() for h in self.humans],
            zombies=[z.copy() for z in self.zombies])


def closest(entities: List[T], position: Vector) -> T:
    return min(entities, key=lambda e: distance2(e.position, position), default=None)


def arg_closest(entities: List[T], position: Vector) -> int:
    min_index = 0
    min_dist = float('inf')
    for i in range(len(entities)):
        d = distance2(entities[i].position, position)
        if d < min_dist:
            min_dist = d
            min_index = i
    return min_index


"""
------------------------------------------------------------------------------------------------------------------------
GAME RULES
------------------------------------------------------------------------------------------------------------------------
"""


def fibs(n: int):
    xs = [0, 1]
    while len(xs) < n:
        xs.append(xs[-2] + xs[-1])
    return xs


FIBONNACCI_NUMBERS = fibs(1000)


def update(game_state: GameState, target: Vector) -> float:
    score = 0.
    nb_killed = 0
    human_factor = 10 * len(game_state.humans) ** 2

    # Move player
    direction = target - game_state.player
    game_state.player += direction / norm(direction) * ASH_SPEED

    # Move zombies / eat humans
    write = 0
    for read in range(len(game_state.zombies)):
        z = game_state.zombies[read]
        z.position = z.next_position
        h_index = arg_closest(game_state.humans, z.position)
        h_position = game_state.humans[h_index].position
        if distance2(z.position, game_state.player) < distance2(z.position, h_position):
            h_position = game_state.player

        direction = h_position - z.position
        direction_norm = norm(direction)
        if direction_norm < ZOMBIE_SPEED:
            z.next_position = h_position
        else:
            z.next_position = np.trunc(z.position + direction / direction_norm * ZOMBIE_SPEED)

        if distance2(z.position, game_state.player) <= SHOOTING_RADIUS ** 2:
            nb_killed += 1
            score += human_factor * FIBONNACCI_NUMBERS[nb_killed+2]
        else:
            game_state.zombies[write] = z
            write += 1
            if np.array_equal(h_position, z.position):
                game_state.humans.pop(h_index)
                if len(game_state.humans) == 0:
                    return score
    game_state.zombies = game_state.zombies[:write]
    return score


"""
------------------------------------------------------------------------------------------------------------------------
AGENT
------------------------------------------------------------------------------------------------------------------------
"""


SEQUENCE_LEN = 4


def can_save_human(zombie_dist, player_dist):
    return (zombie_dist / ZOMBIE_SPEED) > (player_dist / ASH_SPEED - 2)  # Minus the number of turns to shoot


class Agent:
    def __init__(self):
        self.chrono = Chronometer()

    def get_action(self, game_state: GameState) -> Vector:
        self.chrono.start()

        # Monte carlo simulation
        nb_scenario = 0
        closest_human = self.closest_savable_human(game_state)
        best_sequence: List[Vector] = [closest_human.position] * SEQUENCE_LEN
        best_score = self.evaluate(game_state, best_sequence)

        while self.chrono.spent() < 0.9 * MAX_RESPONSE_TIME:
            nb_scenario += 1
            xs = np.random.uniform(0, MAP_WIDTH, size=SEQUENCE_LEN)
            ys = np.random.uniform(0, MAP_HEIGHT, size=SEQUENCE_LEN)
            score = self.evaluate(game_state, zip(xs, ys))
            if score > best_score:
                best_score = score
                best_sequence = list(zip(xs, ys))

        debug("Total time spent", self.chrono.spent(), "ms")
        debug("# Scenario:", nb_scenario)
        debug("# Best score:", best_score)
        return best_sequence[0]

    def evaluate(self, game_state: GameState, sequence):
        score = 0.
        new_state = game_state.clone()
        for action in sequence:
            score += update(new_state, action)
            if not new_state.humans:
                return float('-inf')
            elif not new_state.zombies:
                return score

        h = self.closest_savable_human(new_state)
        if h is None:
            return float('-inf')
        return score - distance(h.position, game_state.player)

    def closest_savable_human(self, game_state: GameState):
        closest_human: Human = None
        closest_dist = float('inf')
        for h in game_state.humans:
            closest_zombie = closest(game_state.zombies, h.position)
            closest_distance = distance(closest_zombie.position, h.position)
            if can_save_human(closest_distance, distance(game_state.player, h.position)):
                d = distance2(h.position, game_state.player)
                if d < closest_dist:
                    closest_dist = d
                    closest_human = h
        return closest_human


'''
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
        closest_human = closest(savable, game_state.player)
        closest_zombie = closest(game_state.zombies, closest_human.position)

        # TODO - try to find if I can get a move that draws zombies to me / go to barycenter of zombies? in some map...
        # TODO - take into account the speed of zombies in order to draw them to you...
        # TODO - try clusterization algorithm to find the "groups" of zombies

        # If the zombie is closer than the human to defend
        if distance2(player_pos, closest_zombie.position) < distance2(player_pos, closest_human.position):

            # If the player is in between, the human is safe for now, try to collect max zombies
            my_zombies = self.zombies_on_player(game_state)
            if len(my_zombies) > 1 and distance2(player_pos, closest_human.position) < distance2(closest_human.position, closest_zombie.position):
                target_position = self.find_attaction_point(player_pos, my_zombies)
                # TODO - once you find the attaction point, plot a path to it that avoids the zombies and keep them in range

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

    def find_attaction_point(self, player_pos: Vector, zombies: List[Zombie]) -> Vector:
        best_point = None
        best_score = float('inf')
        xs = np.random.uniform(0, MAP_WIDTH, size=500)
        ys = np.random.uniform(0, MAP_HEIGHT, size=500)
        for x, y in itertools.chain([player_pos], zip(xs, ys)):
            point = np.array([x, y])
            distances = np.array([distance(z.position, point) for z in zombies])
            variance = np.mean(distances ** 2) - np.mean(distances) ** 2
            score = variance
            if score < best_score:
                best_score = score
                best_point = point
        return best_point
'''


"""
------------------------------------------------------------------------------------------------------------------------
GAME LOOP
------------------------------------------------------------------------------------------------------------------------
"""


def check_prediction(prediction: GameState, game_state: GameState):
    if prediction.zombies != game_state.zombies:
        for ez, gz in zip(prediction.zombies, game_state.zombies):
            if ez != gz:
                debug("EXPECTED")
                debug(ez)
                debug("GOT")
                debug(gz)


def game_loop():
    agent = Agent()
    prediction: GameState = None
    while True:
        game_state = GameState.read()
        if prediction:
            check_prediction(prediction, game_state)
        target = agent.get_action(game_state)
        update(game_state, target)
        prediction = game_state
        print(int(target[0]), int(target[1]), "I love zombies")


if __name__ == '__main__':
    game_loop()
