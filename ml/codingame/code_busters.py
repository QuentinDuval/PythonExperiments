import copy
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

GHOST_MOVE_DISTANCE = 400
MAX_MOVE_DISTANCE = 800

MIN_BUST_DISTANCE = 900
MAX_BUST_DISTANCE = 1760
MAX_STUN_DISTANCE = 1760

STUN_COOLDOWN = 20

TEAM_CORNERS = np.array([[0, 0], [WIDTH, HEIGHT]], dtype=np.float32)


@dataclass(frozen=False)
class Ghost:
    uid: int
    position: np.ndarray
    endurance: int
    busters_count: int = 0
    last_seen: int = 0

    def clone(self):
        return copy.deepcopy(self)


@dataclass(frozen=False)
class Buster:
    uid: int
    position: np.ndarray
    team: int  # 0 for team 0, 1 for team 1
    carried_ghost: int  # ID of the ghost being carried, -1 if no ghost carried
    busting_ghost: bool  # TODO - replace by ID?
    last_seen: int = 0  # Mostly useful for opponent
    stun_duration: int = 0  # How many rounds until end of stun
    stun_cooldown: int = 0

    def clone(self):
        return copy.deepcopy(self)


@dataclass(frozen=False)
class Entities:
    my_team: int
    busters_per_player: int
    ghost_count: int
    busters: Dict[int, Buster] = field(default_factory=dict)
    ghosts: Dict[int, Ghost] = field(default_factory=dict)

    @property
    def his_team(self):
        return 1 - self.my_team

    def get_player_busters(self) -> Iterator[Buster]:
        return (b for _, b in self.busters.items() if b.team == self.my_team)

    def get_opponent_busters(self) -> Iterator[Buster]:
        return (b for _, b in self.busters.items() if b.team != self.my_team)

    def clone(self):
        return copy.deepcopy(self)


"""
------------------------------------------------------------------------------------------------------------------------
READING & TRACKING GAME STATE
------------------------------------------------------------------------------------------------------------------------
"""


def read_entities(my_team_id: int, busters_per_player: int, ghost_count: int, previous_entities: Entities):
    entities = Entities(my_team=my_team_id, busters_per_player=busters_per_player, ghost_count=ghost_count)
    n = int(input())
    for i in range(n):
        # entity_id: buster id or ghost id
        # x, y: position of this buster / ghost
        # entity_type: the team id if it is a buster, -1 if it is a ghost.
        # state:
        #   For busters: 0=idle, 1=carrying a ghost, 2=stunned, 3=busting
        #   For ghosts: endurance
        # value:
        #   For busters: Ghost id being carried, or number of stunned turns if stunned.
        #   For ghosts: number of busters attempting to trap this ghost.
        identity, x, y, entity_type, state, value = [int(j) for j in input().split()]
        position = np.array([x, y])
        if entity_type == -1:
            entities.ghosts[identity] = Ghost(
                uid=identity,
                position=position,
                endurance=state,
                busters_count=value)
        else:
            entities.busters[identity] = Buster(
                uid=identity,
                position=position,
                team=entity_type,
                carried_ghost=value if state == 1 else -1,
                stun_duration=value if state == 2 else 0,
                busting_ghost=state == 3)
    if not previous_entities:
        return entities

    # TODO - beware - ghosts might disappear - make them move for obvious ones
    for ghost_id, ghost in previous_entities.ghosts.items():
        if ghost_id not in entities.ghosts and ghost.last_seen <= 10:  # TODO - parameterize
            entities.ghosts[ghost_id] = ghost

    # TODO - beware these players can move - make them move for obvious ones
    for buster in previous_entities.get_opponent_busters():
        if buster.uid not in entities.busters and buster.last_seen <= 10:  # TODO - parameterize
            entities.busters[buster.uid] = buster
    return entities


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

    def assign_destinations(self, entities: Entities) -> Dict[int, np.ndarray]:
        heap = []
        for buster in entities.get_player_busters():
            for point in self.unvisited:
                d = distance2(point, buster.position)
                x, y = point
                h = self.heat[int(x / self.cell_width), int(y // self.cell_height)]
                heapq.heappush(heap, (d / h, buster.uid, point))

        assignments = {}
        taken = set()
        while heap and len(assignments) < entities.busters_per_player:
            d, player_id, point = heapq.heappop(heap)
            if player_id not in assignments:
                if point not in taken:
                    assignments[player_id] = point
                    taken.add(point)
        return assignments

    def track_explored(self, entities: Entities):
        # TODO - inefficient (look only neighbors)
        for buster in entities.get_player_busters():
            for point in list(self.unvisited):
                if distance2(point, buster.position) < self.cell_dist2:
                    self.unvisited.discard(point)


"""
------------------------------------------------------------------------------------------------------------------------
AGENT
------------------------------------------------------------------------------------------------------------------------
"""


# TODO - different strategies when 2 busters (explore quickly) VS 4 busters (MORE STUNS)
# TODO - different strategies based on number of remaining busters
# TODO - escort remaining buster when it is the end (+ the guy can still STUN!)
# TODO - in the beginning of the game, rush to the center, then bring back ghosts with you?

# TODO - you could wait for the opponent at his base...
# TODO - do not stun when the opponent is in his base
# TODO - stun when the opponent is busting / having a ghost? only if you can STEAL the ghost
# TODO - stick around opponent busters to steal their ghosts
# TODO - have busters that are there to ANNOY and STEAL

# TODO - keep track of previous state to enrich current (and track ghosts moves to help find them)
# TODO - when you find a ghost for the first time (requires tracking): add a "point of interest" to map
# TODO - consider going for a stun if the opponent carries a ghost - pursing him, etc


class Agent:
    def __init__(self):
        self.territory = Territory()
        self.actions: Dict[int, Action] = {}
        self.chrono = Chronometer()

    def get_actions(self, entities: Entities) -> List[Action]:
        self.chrono.start()
        self.actions.clear()
        self.territory.track_explored(entities)

        # TODO - first do an analysis of situation to transition phases?

        self.if_has_ghost_go_to_base(entities)
        self.stun_closest_opponents(entities)
        self.go_fetch_closest_ghosts(entities)
        self.go_explore_territory(entities)
        self.go_to_middle(entities)

        debug("Time spent:", self.chrono.spent())
        return [action for _, action in sorted(self.actions.items(), key=lambda p: p[0])]

    def if_has_ghost_go_to_base(self, entities: Entities):
        for buster in entities.get_player_busters():
            if buster.carried_ghost >= 0:
                player_corner = TEAM_CORNERS[entities.my_team]
                if distance2(buster.position, player_corner) < RADIUS_BASE ** 2:
                    self.actions[buster.uid] = Release()
                    del entities.ghosts[buster.carried_ghost]
                else:
                    self.actions[buster.uid] = Move(player_corner)

    def stun_closest_opponents(self, entities: Entities):
        busters = entities.get_player_busters()
        opponents = entities.get_opponent_busters()
        for opponent in opponents:
            if opponent.stun_duration <= 1:  # 1 allows to stun-lock but avoids double stuns
                for buster in busters:
                    if buster.uid not in self.actions and buster.stun_cooldown <= 0:
                        if distance2(buster.position, opponent.position) < MAX_STUN_DISTANCE ** 2:
                            self.actions[buster.uid] = Stun(opponent.uid)
                            entities.stun_cooldown = STUN_COOLDOWN
                        elif distance2(buster.position, opponent.position) < RADIUS_SIGHT ** 2:
                            self.actions[buster.uid] = Move(opponent.position)

    def go_fetch_closest_ghosts(self, entities: Entities):
        if not entities.ghosts:
            return

        # Prioritize the ghosts with low endurance - TODO - take into account distance + distance to base as well => mix
        ghosts: List[Ghost] = list(entities.ghosts.values())
        ghosts.sort(key=lambda g: g.endurance)
        busters: List[Buster] = list(entities.get_player_busters())

        debug(ghosts)
        debug(busters)

        for ghost in ghosts:
            busters.sort(key=lambda b: distance2(ghost.position, b.position))

            # TODO - take into account move of ghosts to choose where to go
            # TODO - compute a number of busters to assign? Should instead have a kind of priority
            for buster in busters:
                if buster.uid not in self.actions:
                    closest_dist2 = distance2(ghost.position, buster.position)
                    if closest_dist2 == 0:
                        next_player_pos = np.array([WIDTH / 2, HEIGHT / 2])
                        self.actions[buster.uid] = Move(next_player_pos)
                    elif closest_dist2 < MIN_BUST_DISTANCE ** 2:
                        next_player_dir = buster.position - ghost.position
                        next_player_pos = buster.position + next_player_dir / norm(next_player_dir) * (
                                    MAX_MOVE_DISTANCE - GHOST_MOVE_DISTANCE)
                        self.actions[buster.uid] = Move(next_player_pos)
                    elif closest_dist2 < MAX_BUST_DISTANCE ** 2:
                        self.actions[buster.uid] = Bust(ghost.uid)
                    else:
                        self.actions[buster.uid] = Move(ghost.position)

                    # Do not put too many busters on a weak ghost
                    if ghost.endurance <= 5:
                        break

    def go_explore_territory(self, entities: Entities):
        assignments = self.territory.assign_destinations(entities)
        for player_id, point in assignments.items():
            if player_id not in self.actions:
                self.actions[player_id] = Move(point)

    def go_to_middle(self, entities: Entities):
        for buster in entities.get_player_busters():
            if buster.uid not in self.actions:
                self.actions[buster.uid] = Move(np.ndarray([8000, 4500]))


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
    previous_entities = None
    while True:
        entities = read_entities(my_team_id=my_team_id,
                                 busters_per_player=busters_per_player,
                                 ghost_count=ghost_count,
                                 previous_entities=previous_entities)
        debug(entities)

        actions = agent.get_actions(entities)
        for action in actions:
            print(action)

        previous_entities = entities


if __name__ == '__main__':
    game_loop()
