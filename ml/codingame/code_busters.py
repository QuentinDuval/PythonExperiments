from dataclasses import *
import math
import sys
import itertools
from typing import *
import time

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
    return v[0] ** 2 + v[1] ** 2


def norm(v) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2)


def distance2(from_, to_) -> float:
    return (to_[0] - from_[0]) ** 2 + (to_[1] - from_[1]) ** 2


def distance(from_, to_) -> float:
    return math.sqrt(distance2(from_, to_))


def mod_angle(angle: Angle) -> Angle:
    if angle > 2 * math.pi:
        return angle - 2 * math.pi
    if angle < 0:
        return angle + 2 * math.pi
    return angle


"""
------------------------------------------------------------------------------------------------------------------------
MAIN DATA STRUCTURE
------------------------------------------------------------------------------------------------------------------------
"""

WIDTH = 16000
HEIGHT = 9000

RADIUS_BASE = 1600
RADIUS_SIGHT = 2200

MAX_MOVE_DISTANCE = 800
MIN_BUST_DISTANCE = 900
MAX_BUST_DISTANCE = 1760

TEAM_CORNERS = np.array([[0, 0], [WIDTH, HEIGHT]], dtype=np.float32)


# TODO - player and ghosts can have the SAME IDs


@dataclass(frozen=False)
class Entities:
    my_team: int
    busters_per_player: int
    ghost_count: int
    position: np.ndarray
    entity_type: np.ndarray  # entity_type: the team id if it is a buster, -1 if it is a ghost.
    carrying: np.ndarray  # state: For busters: 0=idle, 1=carrying a ghost.
    value: np.ndarray  # For busters: Ghost id being carried. For ghosts: number of busters attempting to trap this ghost.
    valid: np.ndarray  # Whether a ghost is still on the map - TODO: have a probability of the ghost being there still

    @property
    def his_team(self):
        return 1 - self.my_team

    def __len__(self):
        return self.valid.shape[0]

    @classmethod
    def empty(cls, my_team: int, busters_per_player: int, ghost_count: int):
        size = busters_per_player * 2 + ghost_count
        return Entities(
            my_team=my_team,
            busters_per_player=busters_per_player,
            ghost_count=ghost_count,
            position=np.zeros(shape=(size, 2), dtype=np.float32),
            entity_type=np.full(shape=size, fill_value=-1, dtype=np.int8),
            carrying=np.zeros(shape=size, dtype=bool),
            value=np.zeros(shape=size, dtype=np.int8),
            valid=np.full(shape=size, fill_value=False, dtype=np.int8)
        )


"""
------------------------------------------------------------------------------------------------------------------------
AGENT
------------------------------------------------------------------------------------------------------------------------
"""

AREA_SPOTS = np.array([
    [WIDTH / 2, HEIGHT / 2],
    [WIDTH / 2, 0.],
    [WIDTH / 2, HEIGHT]
], dtype=np.float32)


class Agent:
    def __init__(self):
        pass

    def get_actions(self, entities: Entities) -> List[str]:
        # Possible actions: MOVE x y | BUST id | RELEASE

        ghost_ids = self.get_ghost_ids(entities)
        player_ids = self.get_player_ids(entities)
        invisible_ids = self.get_invisible_area_ids(entities, player_ids)

        debug(entities)
        debug("player ids:", player_ids)
        debug("ghost ids:", ghost_ids)
        debug("invisible ids:", invisible_ids)

        for player_id in player_ids:
            player_pos = entities.position[player_id]

            # If has ghost => go to base to release it
            if entities.carrying[player_id]:
                debug("bringing ghost", player_id)
                player_corner = TEAM_CORNERS[entities.my_team]
                if distance2(player_pos, player_corner) < RADIUS_BASE ** 2:
                    released_ghost_id = entities.value[player_id]
                    entities.valid[released_ghost_id] = False
                    yield "RELEASE"
                else:
                    yield "MOVE " + str(int(player_corner[0])) + " " + str(int(player_corner[1]))

            # Go fetch the closest ghost
            elif ghost_ids:
                debug("capture ghost", player_id)
                closest_id = min(ghost_ids, key=lambda gid: distance2(entities.position[gid], player_pos))
                closest_dist2 = distance2(player_pos, entities.position[closest_id])
                ghost_ids.remove(closest_id)
                if closest_dist2 < MAX_BUST_DISTANCE ** 2:
                    yield "BUST " + str(closest_id)
                else:
                    x, y = entities.position[closest_id]
                    yield "MOVE " + str(int(x)) + " " + str(int(y))

            # Try to find some ghosts
            elif invisible_ids:
                debug("searching for ghosts", player_id)
                closest_id = min(invisible_ids, key=lambda aid: distance2(AREA_SPOTS[aid], player_pos))
                x, y = AREA_SPOTS[closest_id]
                invisible_ids.remove(closest_id)
                yield "MOVE " + str(int(x)) + " " + str(int(y))

            # Do nothing...
            else:
                debug("wandering", player_id)
                yield "MOVE 8000 4500"

    def get_player_ids(self, entities: Entities) -> List[int]:
        ids = []
        for i in range(len(entities)):
            if entities.entity_type[i] == entities.my_team:
                ids.append(i)
        return ids

    def get_ghost_ids(self, entities: Entities) -> Set[int]:
        ids = set()
        for i in range(len(entities)):
            if entities.entity_type[i] == -1:
                if entities.valid[i]:
                    ids.add(i)
        return ids

    def get_invisible_area_ids(self, entities: Entities, player_ids: List[int]) -> Set[int]:
        invisible_ids = set(range(len(AREA_SPOTS)))
        for player_id in player_ids:
            for invisible_id in list(invisible_ids):
                if distance2(entities.position[player_id], AREA_SPOTS[invisible_id]) < RADIUS_SIGHT ** 2:
                    invisible_ids.remove(invisible_id)
        return invisible_ids


"""
------------------------------------------------------------------------------------------------------------------------
GAME LOOP
------------------------------------------------------------------------------------------------------------------------
"""


def read_entities(entities: Entities):
    n = int(input())
    for i in range(n):
        entity_id, x, y, entity_type, state, value = [int(j) for j in input().split()]
        debug(entity_id, x, y, entity_type, state, value)
        entities.position[entity_id][0] = x
        entities.position[entity_id][1] = y
        entities.entity_type[entity_id] = entity_type
        entities.carrying[entity_id] = state
        entities.value[entity_id] = value
        entities.valid[entity_id] = True


def game_loop():
    busters_per_player = int(input())
    ghost_count = int(input())
    my_team_id = int(input())

    debug("buster by player: ", busters_per_player)
    debug("ghost count: ", ghost_count)
    debug("my team id: ", my_team_id)

    entities = Entities.empty(my_team=my_team_id, busters_per_player=busters_per_player, ghost_count=ghost_count)
    agent = Agent()
    while True:
        read_entities(entities)
        for action in agent.get_actions(entities):
            print(action)


if __name__ == '__main__':
    game_loop()
