from dataclasses import *
import math
import sys
import itertools
from typing import *
import time
import heapq

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


class Stun(NamedTuple):
    buster_id: int

    def __repr__(self):
        return "STUN " + str(self.buster_id)


Action = Union[Move, Bust, Release, Stun]


"""
------------------------------------------------------------------------------------------------------------------------
MAIN DATA STRUCTURE to keep STATE
------------------------------------------------------------------------------------------------------------------------
"""


WIDTH = 16000
HEIGHT = 9000

RADIUS_BASE = 1600
RADIUS_SIGHT = 2200

MAX_MOVE_DISTANCE = 800
MIN_BUST_DISTANCE = 900
MAX_BUST_DISTANCE = 1760
MAX_STUN_DISTANCE = 1760

STUN_COOLDOWN = 20

TEAM_CORNERS = np.array([[0, 0], [WIDTH, HEIGHT]], dtype=np.float32)


@dataclass(frozen=False)
class Entities:
    my_team: int
    busters_per_player: int
    ghost_count: int

    buster_position: np.ndarray
    buster_team: np.ndarray     # 0 for team 0, 1 for team 1
    buster_ghost: np.ndarray    # ID of the ghost being carried, -1 if no ghost carried
    buster_stunned: np.ndarray
    buster_busting: np.ndarray
    buster_cooldown: np.ndarray

    ghost_position: np.ndarray
    ghost_attempt: np.ndarray   # Number of busters attempting to trap this ghost.
    ghost_endurance: np.ndarray
    ghost_valid: np.ndarray  # Whether a ghost is on the map - TODO: have a probability

    @property
    def his_team(self):
        return 1 - self.my_team

    @classmethod
    def empty(cls, my_team: int, busters_per_player: int, ghost_count: int):
        return Entities(
            my_team=my_team,
            busters_per_player=busters_per_player,
            ghost_count=ghost_count,

            buster_position=np.zeros(shape=(busters_per_player * 2, 2), dtype=np.float32),
            buster_team=np.full(shape=busters_per_player * 2, fill_value=-1, dtype=np.int8),
            buster_ghost=np.full(shape=busters_per_player * 2, fill_value=-1, dtype=np.int8),
            buster_stunned=np.zeros(shape=busters_per_player * 2, dtype=np.int8),
            buster_busting=np.full(shape=busters_per_player * 2, fill_value=False, dtype=bool),
            buster_cooldown=np.zeros(shape=busters_per_player * 2, dtype=np.int8),

            ghost_position=np.zeros(shape=(ghost_count, 2), dtype=np.float32),
            ghost_attempt=np.zeros(shape=ghost_count, dtype=np.int8),
            ghost_endurance=np.zeros(shape=ghost_count, dtype=np.int8),
            ghost_valid=np.full(shape=ghost_count, fill_value=False, dtype=bool)
        )

    def get_player_ids(self) -> List[int]:
        ids = []
        for i in range(self.buster_team.shape[0]):
            if self.buster_team[i] == self.my_team:
                ids.append(i)
        return ids

    def get_opponent_ids(self) -> List[int]:
        ids = []
        for i in range(self.buster_team.shape[0]):
            if self.buster_team[i] >= 0 and self.buster_team[i] != self.my_team:
                ids.append(i)
        return ids

    def get_ghost_ids(self) -> Set[int]:
        ids = set()
        for i in range(self.ghost_count):
            if self.ghost_valid[i]:
                ids.add(i)
        return ids


def read_entities(my_team_id: int, busters_per_player: int, ghost_count: int):
    entities = Entities.empty(my_team=my_team_id, busters_per_player=busters_per_player, ghost_count=ghost_count)
    n = int(input())
    for i in range(n):
        # entity_id: buster id or ghost id
        # x, y: position of this buster / ghost
        # entity_type: the team id if it is a buster, -1 if it is a ghost.
        # state:
        #   For busters: 0=idle, 1=carrying a ghost, 2=stunned, 3=busting
        #   For ghosts: endurance
        # value:
        #   For busters: Ghost id being carried.
        #   For ghosts: number of busters attempting to trap this ghost.
        identity, x, y, entity_type, state, value = [int(j) for j in input().split()]
        if entity_type == -1:
            entities.ghost_position[identity][0] = x
            entities.ghost_position[identity][1] = y
            entities.ghost_endurance[identity] = state
            entities.ghost_attempt[identity] = value
            entities.ghost_valid[identity] = True
        else:
            entities.buster_position[identity][0] = x
            entities.buster_position[identity][1] = y
            entities.buster_team[identity] = entity_type
            entities.buster_ghost[identity] = value if state == 1 else -1
            entities.buster_stunned[identity] = value if state == 2 else 0
            entities.buster_busting[identity] = state == 3
    return entities


"""
------------------------------------------------------------------------------------------------------------------------
TRACKING GAME STATE between turns
------------------------------------------------------------------------------------------------------------------------
"""


@dataclass(frozen=False)
class GameState:
    stun_cooldown: Dict[int, int] = field(default_factory=dict)

    def new_turn(self):
        for player_id, cooldown in list(self.stun_cooldown.items()):
            if cooldown > 1:
                self.stun_cooldown[player_id] = cooldown - 1
            else:
                del self.stun_cooldown[player_id]

    def enrich_state(self, entities: Entities):
        for player_id, cooldown in self.stun_cooldown.items():
            entities.buster_cooldown[player_id] = cooldown

    def on_actions(self, entities: Entities, actions: List[Action]):
        for player_id, action in zip(entities.get_player_ids(), actions):
            if isinstance(action, Stun):
                self.stun_cooldown[player_id] = STUN_COOLDOWN + 1


"""
------------------------------------------------------------------------------------------------------------------------
TERRITORY - TO GUIDE EXPLORATION
------------------------------------------------------------------------------------------------------------------------
"""


class Territory:
    def __init__(self, w=15, h=10):
        self.unvisited = set()
        self.w = w
        self.h = h
        self.cell_width = WIDTH / self.w
        self.cell_height = HEIGHT / self.h
        self.cell_dist2 = norm2([self.cell_width, self.cell_height]) / 2
        for i in range(self.w):
            for j in range(self.h):
                x = self.cell_width / 2 + self.cell_width * i
                y = self.cell_height / 2 + self.cell_height * j
                self.unvisited.add((x, y))

        center = (self.w / 2, self.h / 2)
        self.heat = np.ones(shape=(self.w, self.h), dtype=np.float32)  # TODO: use better encoding of territory (array?)
        for i in range(self.w):
            for j in range(self.h):
                self.heat[(i, j)] = 10 / (1 + distance((i, j), center))

    def __len__(self):
        return len(self.unvisited)

    def assign_destinations(self, entities: Entities, player_ids: List[int]) -> Dict[int, np.ndarray]:
        # TODO - there is a notion of "not so useful to explore territory" => heat map to do
        # TODO - there is a notion of "point of interest" that should impact the heat map (ex: probable ghost here)

        heap = []
        for player_id in player_ids:
            player_pos = entities.buster_position[player_id]
            for point in self.unvisited:
                d = distance2(point, player_pos)
                x, y = point
                h = self.heat[int(x / self.cell_width), int(y // self.cell_height)]
                heapq.heappush(heap, (d / h, player_id, point))

        assignments = {}
        taken = set()
        while heap and len(assignments) < len(player_ids):
            d, player_id, point = heapq.heappop(heap)
            if player_id not in assignments:
                if point not in taken:
                    assignments[player_id] = point
                    taken.add(point)
        return assignments

    def track_explored(self, entities: Entities, player_ids: List[int]):
        for player_id in player_ids:
            player_pos = entities.buster_position[player_id]
            for point in list(self.unvisited):  # TODO - make it more efficient (only neighbors)
                if distance2(point, player_pos) < self.cell_dist2:
                    self.unvisited.discard(point)


"""
------------------------------------------------------------------------------------------------------------------------
AGENT
------------------------------------------------------------------------------------------------------------------------
"""


class Agent:
    def __init__(self):
        self.territory = Territory()
        self.actions = {}
        self.chrono = Chronometer()

    def get_actions(self, entities: Entities) -> List[Action]:
        # TODO - different strategies when 2 busters (explore quickly) VS 4 busters (MORE STUNS)

        # TODO - several busters by ghost for ghosts with high endurance
        # TODO - stun when the opponent is busting / having a ghost? only if you can STEAL the ghost
        # TODO - stick around opponent busters to steal their ghosts
        # TODO - have busters that are there to ANNOY and STEAL

        # TODO - keep track of previous state to enrich current (and track ghosts moves to help find them)
        # TODO - when you find a ghost for the first time (requires tracking): add a "point of interest" to map
        # TODO - consider going for a stun if the opponent carries a ghost - pursing him, etc

        self.chrono.start()
        self.actions.clear()
        ghost_ids = entities.get_ghost_ids()
        player_ids = entities.get_player_ids()
        self.territory.track_explored(entities, player_ids)

        debug("player ids:", player_ids)
        debug("ghost ids:", ghost_ids)

        self.if_has_ghost_go_to_base(entities, player_ids)
        self.go_fetch_closest_ghosts(entities, player_ids)
        self.stun_closest_opponents(entities, player_ids)
        self.go_explore_territory(entities, player_ids)
        self.go_to_middle(entities, player_ids)

        debug("Time spent:", self.chrono.spent())
        return [self.actions[player_id] for player_id in player_ids]

    def if_has_ghost_go_to_base(self, entities: Entities, player_ids: List[int]):
        for player_id in player_ids:
            if entities.buster_ghost[player_id] >= 0:
                player_corner = TEAM_CORNERS[entities.my_team]
                player_pos = entities.buster_position[player_id]
                if distance2(player_pos, player_corner) < RADIUS_BASE ** 2:
                    self.actions[player_id] = Release()
                else:
                    self.actions[player_id] = Move(player_corner)

    def go_fetch_closest_ghosts(self, entities: Entities, player_ids: List[int]):
        ghost_ids = entities.get_ghost_ids()
        if not ghost_ids:
            return

        player_ids = list(player_ids)
        for ghost_id in ghost_ids:
            ghost_pos = entities.ghost_position[ghost_id]
            player_ids.sort(key=lambda b: distance2(ghost_pos, entities.buster_position[b]))
            for player_id in player_ids:
                if player_id not in self.actions:
                    closest_dist2 = distance2(ghost_pos, entities.buster_position[player_id])
                    if closest_dist2 < MAX_BUST_DISTANCE ** 2:
                        self.actions[player_id] = Bust(ghost_id)
                    else:
                        self.actions[player_id] = Move(ghost_pos)
                    break

    def stun_closest_opponents(self, entities: Entities, player_ids: List[int]):
        opponent_ids = entities.get_opponent_ids()
        for opponent_id in opponent_ids:
            opponent_pos = entities.buster_position[opponent_id]
            for player_id in player_ids:
                if player_id not in self.actions and entities.buster_cooldown[player_id] <= 0:
                    player_pos = entities.buster_position[player_id]
                    if distance2(player_pos, opponent_pos) < MAX_STUN_DISTANCE ** 2:
                        self.actions[player_id] = Stun(opponent_id)

    def go_explore_territory(self, entities: Entities, player_ids: List[int]):
        player_ids = [player_id for player_id in player_ids if player_id not in self.actions]
        assignments = self.territory.assign_destinations(entities, player_ids)
        for player_id, point in assignments.items():
            self.actions[player_id] = Move(point)

    def go_to_middle(self, entities: Entities, player_ids: List[int]):
        for player_id in player_ids:
            if player_id not in self.actions:
                self.actions[player_id] = Move(np.ndarray([8000, 4500]))


"""
------------------------------------------------------------------------------------------------------------------------
GAME LOOP
------------------------------------------------------------------------------------------------------------------------
"""


def game_loop():
    busters_per_player = int(input())
    ghost_count = int(input())
    my_team_id = int(input())

    debug("buster by player: ", busters_per_player)
    debug("ghost count: ", ghost_count)
    debug("my team id: ", my_team_id)

    agent = Agent()
    game_state = GameState()
    while True:
        game_state.new_turn()
        entities = read_entities(my_team_id=my_team_id, busters_per_player=busters_per_player, ghost_count=ghost_count)
        game_state.enrich_state(entities)
        actions = agent.get_actions(entities)
        game_state.on_actions(entities, actions)
        for action in actions:
            print(action)


if __name__ == '__main__':
    game_loop()
