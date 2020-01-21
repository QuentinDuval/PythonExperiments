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


@dataclass(frozen=False)
class Entities:
    my_team: int
    busters_per_player: int
    ghost_count: int

    buster_position: np.ndarray
    buster_team: np.ndarray  # 0 for team 0, 1 for team 1
    buster_ghost: np.ndarray  # ID of the ghost being carried, -1 if no ghost carried

    ghost_position: np.ndarray
    ghost_attempt: np.ndarray  # For ghosts: number of busters attempting to trap this ghost.
    ghost_valid: np.ndarray  # Whether a ghost is on the map - TODO: have a probability

    @property
    def his_team(self):
        return 1 - self.my_team

    @classmethod
    def empty(cls, my_team: int, busters_per_player: int, ghost_count: int):
        size = busters_per_player * 2 + ghost_count
        return Entities(
            my_team=my_team,
            busters_per_player=busters_per_player,
            ghost_count=ghost_count,

            buster_position=np.zeros(shape=(busters_per_player * 2, 2), dtype=np.float32),
            buster_team=np.full(shape=busters_per_player * 2, fill_value=-1, dtype=np.int8),
            buster_ghost=np.full(shape=busters_per_player * 2, fill_value=-1, dtype=np.int8),

            ghost_position=np.zeros(shape=(ghost_count, 2), dtype=np.float32),
            ghost_attempt=np.zeros(shape=ghost_count, dtype=np.int8),
            ghost_valid=np.full(shape=size, fill_value=False, dtype=bool)
        )

    def get_player_ids(self) -> List[int]:
        ids = []
        for i in range(self.buster_team.shape[0]):
            if self.buster_team[i] == self.my_team:
                ids.append(i)
        return ids

    def get_ghost_ids(self) -> Set[int]:
        ids = set()
        for i in range(self.ghost_count):
            if self.ghost_valid[i]:
                ids.add(i)
        return ids


"""
------------------------------------------------------------------------------------------------------------------------
ACTION
------------------------------------------------------------------------------------------------------------------------
"""


class Move(NamedTuple):
    position: np.ndarray

    def __repr__(self):
        x, y = self.position
        return "MOVE " + str(int(x)) + " " + str(int(y))


class Bust(NamedTuple):
    ghost_id: int

    def __repr__(self):
        return "BUST " + str(self.ghost_id)


class Release:
    def __repr__(self):
        return "RELEASE"


Action = Union[Move, Bust, Release]


"""
------------------------------------------------------------------------------------------------------------------------
AGENT
------------------------------------------------------------------------------------------------------------------------
"""


class Territory:
    w = 15
    h = 10

    def __init__(self):
        self.unvisited = set()
        self.cell_width = WIDTH / self.w
        self.cell_height = HEIGHT / self.h
        self.cell_dist2 = norm2([self.cell_width, self.cell_height]) / 2
        for i in range(self.w):
            for j in range(self.h):
                x = self.cell_width / 2 + self.cell_width * i
                y = self.cell_height / 2 + self.cell_height * j
                self.unvisited.add((x, y))

    def __len__(self):
        return len(self.unvisited)

    def assign_destinations(self, entities: Entities, player_ids: List[int]) -> List[np.ndarray]:
        pass

    def shortest_point_to(self, position: np.ndarray) -> np.ndarray:
        closest = None
        min_dist = float('inf')
        for pos in self.unvisited:
            if distance2(pos, position) < min_dist:
                min_dist = distance2(pos, position) + np.random.uniform(0, 3000) ** 2
                closest = pos
        x, y = closest
        return np.array([x, y])

    def track_explored(self, entities: Entities, player_ids: List[int]):
        for player_id in player_ids:
            player_pos = entities.buster_position[player_id]
            for point in list(self.unvisited):                        # TODO - make it more efficient (only neighbors)
                if distance2(point, player_pos) < self.cell_dist2:
                    self.unvisited.discard(point)

    def remove_position(self, position: np.ndarray):
        x, y = position
        self.unvisited.discard((x, y))  # does not crash if key is not there


class Agent:
    def __init__(self):
        self.territory = Territory()
        self.assignment = {}

    def get_actions(self, entities: Entities) -> List[Action]:
        # TODO - keep track of previous state to enrich current (and track ghosts moves)

        ghost_ids = entities.get_ghost_ids()
        player_ids = entities.get_player_ids()
        debug("player ids:", player_ids)
        debug("ghost ids:", ghost_ids)

        self.territory.track_explored(entities, player_ids)
        actions: List[Action] = [None] * len(player_ids)

        # TODO - Look for opportunities to STUNs opponents - and store cooldown + status of opponent
        # TODO - separate in several phases (to avoid conflicts and priority of busters)
        self.if_has_ghost_go_to_base(entities, player_ids, actions)
        self.go_fetch_closest_ghosts(entities, player_ids, actions)
        self.go_explore_territory(entities, player_ids, actions)
        return actions

    def if_has_ghost_go_to_base(self, entities: Entities, player_ids: List[int], actions: List[Action]):
        for i, player_id in enumerate(player_ids):
            if entities.buster_ghost[player_id] >= 0:
                player_corner = TEAM_CORNERS[entities.my_team]
                player_pos = entities.buster_position[player_id]
                if distance2(player_pos, player_corner) < RADIUS_BASE ** 2:
                    entities.ghost_valid[entities.buster_ghost[player_id]] = False   # TODO - rework the tracking
                    actions[i] = Release()
                else:
                    actions[i] = Move(player_corner)

    def go_fetch_closest_ghosts(self, entities: Entities, player_ids: List[int], actions: List[Action]):
        # TODO - use correct priority queues to go to the closest
        ghost_ids = entities.get_ghost_ids()
        for i, player_id in enumerate(player_ids):
            if actions[i] is None:
                if ghost_ids:
                    player_pos = entities.buster_position[player_id]
                    closest_id = min(ghost_ids, key=lambda gid: distance2(entities.ghost_position[gid], player_pos))
                    closest_dist2 = distance2(player_pos, entities.ghost_position[closest_id])
                    ghost_ids.remove(closest_id)
                    if closest_dist2 < MAX_BUST_DISTANCE ** 2:
                        actions[player_id] = Bust(closest_id)
                    else:
                        actions[player_id] = Move(entities.ghost_position[closest_id])

    def go_explore_territory(self, entities: Entities, player_ids: List[int], actions: List[Action]):
        for i, player_id in enumerate(player_ids):
            if actions[i] is None:

                player_pos = entities.buster_position[player_id]

                # If the player has some area to explore
                if player_id in self.assignment:
                    debug("following assignment", player_id)
                    point = self.assignment[player_id]
                    if distance2(point, player_pos) < self.territory.cell_dist2:
                        del self.assignment[player_id]
                        self.territory.remove_position(point)
                    actions[i] = Move(point)

                # Try to find some ghosts
                elif self.territory:
                    debug("exploring territory", player_id)
                    point = self.territory.shortest_point_to(entities.buster_position[player_id])
                    self.assignment[player_id] = point
                    actions[i] = Move(point)

                # Do nothing...
                else:
                    actions[i] = Move(np.ndarray([8000, 4500]))

"""
------------------------------------------------------------------------------------------------------------------------
GAME LOOP
------------------------------------------------------------------------------------------------------------------------
"""


def read_entities(my_team_id: int, busters_per_player: int, ghost_count: int):
    entities = Entities.empty(my_team=my_team_id, busters_per_player=busters_per_player, ghost_count=ghost_count)
    n = int(input())
    for i in range(n):
        # entity_id: buster id or ghost id
        # x, y: position of this buster / ghost
        # entity_type: the team id if it is a buster, -1 if it is a ghost.
        # state: For busters: 0=idle, 1=carrying a ghost.
        # value: For busters: Ghost id being carried. For ghosts: number of busters attempting to trap this ghost.
        entity_id, x, y, entity_type, state, value = [int(j) for j in input().split()]
        if entity_type == -1:
            entities.ghost_position[entity_id][0] = x
            entities.ghost_position[entity_id][1] = y
            entities.ghost_attempt[entity_id] = value
            entities.ghost_valid[entity_id] = True
        else:
            entities.buster_position[entity_id][0] = x
            entities.buster_position[entity_id][1] = y
            entities.buster_team[entity_id] = entity_type
            entities.buster_ghost[entity_id] = -1 if state == 0 else value
    return entities


def game_loop():
    busters_per_player = int(input())
    ghost_count = int(input())
    my_team_id = int(input())

    debug("buster by player: ", busters_per_player)
    debug("ghost count: ", ghost_count)
    debug("my team id: ", my_team_id)

    agent = Agent()
    while True:
        entities = read_entities(my_team_id=my_team_id, busters_per_player=busters_per_player, ghost_count=ghost_count)
        for action in agent.get_actions(entities):
            print(action)


if __name__ == '__main__':
    game_loop()