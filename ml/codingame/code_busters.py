import abc
import copy
import enum
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
MAIN DATA STRUCTURE to keep STATE
------------------------------------------------------------------------------------------------------------------------
"""

WIDTH = 16000
HEIGHT = 9000

RADIUS_BASE = 1600
RADIUS_SIGHT = 2200

MAX_GHOST_MOVE_DISTANCE = 400
MAX_MOVE_DISTANCE = 800

MIN_BUST_DISTANCE = 900
MAX_BUST_DISTANCE = 1760
MAX_STUN_DISTANCE = 1760

STUN_DURATION = 10
STUN_COOLDOWN = 20

TEAM_CORNERS = np.array([[0, 0], [WIDTH, HEIGHT]], dtype=np.float32)


EntityId = int;


@dataclass(frozen=False)
class Ghost:
    uid: EntityId
    position: np.ndarray
    endurance: int
    busters_count: int = 0
    last_seen: int = 0

    def clone(self):
        return copy.deepcopy(self)


@dataclass(frozen=False)
class Buster:
    uid: EntityId
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
    my_score: int
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
    entities = Entities(my_team=my_team_id, my_score=0, busters_per_player=busters_per_player, ghost_count=ghost_count)
    carried_ghosts = set()
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
            carried_ghosts.add(value if state == 1 else -1)

    # IMPORTANT NODE:
    # * the carried ghosts are not part of the input

    if not previous_entities:
        return entities

    # Adding the previous information
    entities.my_score = previous_entities.my_score
    for buster in entities.get_player_busters():
        buster.stun_cooldown = previous_entities.busters[buster.uid].stun_cooldown
        # TODO - probably missing stuff here

    # Adding missing ghosts
    for ghost_id, ghost in previous_entities.ghosts.items():
        if ghost_id not in entities.ghosts and ghost_id not in carried_ghosts and ghost.last_seen <= 10:  # TODO - parameterize
            entities.ghosts[ghost_id] = ghost
            ghost.last_seen += 1

    # Adding missing busters
    for buster in previous_entities.get_opponent_busters():
        if buster.uid not in entities.busters and buster.last_seen <= 10:  # TODO - parameterize
            entities.busters[buster.uid] = buster
            buster.last_seen += 1
    return entities


def on_end_of_turn(entities: Entities):
    # TODO - make ghosts move - or make it as ACTION
    # TODO - make opponent busters carrying ghosts move - or make it as ACTION
    # TODO - decrease all the cooldowns
    pass


"""
------------------------------------------------------------------------------------------------------------------------
ACTION
------------------------------------------------------------------------------------------------------------------------
"""


class Move(NamedTuple):
    caster_id: int
    position: np.ndarray

    def apply(self, entities: Entities):
        pass  # TODO - no need now

    def __repr__(self):
        x, y = self.position
        return "MOVE " + str(int(x)) + " " + str(int(y))


class Bust(NamedTuple):
    caster_id: int
    target_id: int

    def apply(self, entities: Entities):
        pass  # TODO - no need now

    def __repr__(self):
        return "BUST " + str(self.target_id)


class Release(NamedTuple):
    caster_id: int
    target_id: int

    def apply(self, entities: Entities):
        entities.busters[self.caster_id].carried_ghost = -1
        del entities.ghosts[self.target_id]

    def __repr__(self):
        return "RELEASE"


class Stun(NamedTuple):
    caster_id: int
    target_id: int

    def apply(self, entities: Entities):
        entities.busters[self.caster_id].stun_cooldown = STUN_COOLDOWN
        entities.busters[self.target_id].stun_duration = STUN_DURATION

    def __repr__(self):
        return "STUN " + str(self.target_buster_id)


Action = Union[Move, Bust, Release, Stun]


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

    def assign_destinations(self, busters: Collection[Buster]) -> Dict[int, Vector]:
        heap = []
        for buster in busters:
            for point in self.unvisited:
                d = distance2(point, buster.position)
                x, y = point
                h = self.heat[int(x / self.cell_width), int(y // self.cell_height)]
                heapq.heappush(heap, (d / h, buster.uid, point))

        assignments = {}
        taken = set()
        while heap and len(assignments) < len(busters):
            d, player_id, point = heapq.heappop(heap)
            if player_id not in assignments:
                if point not in taken:
                    assignments[player_id] = point
                    taken.add(point)
        return assignments

    def track_explored(self, busters: Iterator[Buster]):
        # TODO - inefficient (look only neighbors)
        for buster in busters:
            for point in list(self.unvisited):
                if distance2(point, buster.position) < self.cell_dist2:
                    self.unvisited.discard(point)

    def is_visited(self, point: Vector) -> bool:
        return point not in self.unvisited


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


@dataclass(frozen=False)
class Exploring:
    destination: Vector


@dataclass(frozen=False)
class Capturing:
    target_id: EntityId


@dataclass(frozen=False)
class Carrying:
    pass


@dataclass(frozen=False)
class Escorting:
    pass


@dataclass(frozen=False)
class Intercepting:
    pass


@dataclass(frozen=False)
class Herding:
    pass


# TODO - add a state for the game as well (exploration phase, mid-game, end-game)...


class Agent:
    def __init__(self):
        self.chrono = Chronometer()
        self.territory = Territory()
        self.unassigned: Set[int] = set()
        self.exploring: Dict[int, Exploring] = {}
        self.capturing: Dict[int, Capturing] = {}
        self.carrying: Dict[int, Carrying] = {}

    def init(self, entities: Entities):
        busters = list(entities.get_player_busters())
        assignments = self.territory.assign_destinations(busters)
        for buster in entities.get_player_busters():
            self.exploring[buster.uid] = Exploring(assignments[buster.uid])

    def get_actions(self, entities: Entities) -> List[Action]:
        self.chrono.start()

        self._update_past_states(entities)
        self._state_transitions(entities)
        actions = self._carry_actions(entities)

        debug("Time spent:", self.chrono.spent())
        return [p[1] for p in sorted(actions.items(), key=lambda p: p[0])]

    def _update_past_states(self, entities: Entities):

        # Tracking exploration state
        self.territory.track_explored(entities.get_player_busters())
        for buster_id, exploring in list(self.exploring.items()):
            if self.territory.is_visited(exploring.destination):
                del self.exploring[buster_id]
                self.unassigned.add(buster_id)

        # Carrying update
        for buster_id, carrying in list(self.carrying.items()):
            if entities.busters[buster_id].carried_ghost < 0:
                del self.carrying[buster_id]
                self.unassigned.add(buster_id)

        # Capturing update
        for buster_id, capturing in list(self.capturing.items()):
            if entities.busters[buster_id].carried_ghost >= 0:
                del self.capturing[buster_id]
                self.carrying[buster_id] = Carrying()
            elif not entities.busters[buster_id].busting_ghost:
                del self.capturing[buster_id]
                self.unassigned.add(buster_id)

    def _carry_actions(self, entities: Entities) -> Dict[int, Action]:
        actions = dict()

        # Exploration of the map
        for buster_id, exploring in self.exploring.items():
            actions[buster_id] = Move(buster_id, exploring.destination)

        # Capturing a ghost
        for buster_id, capturing in self.capturing.items():
            buster = entities.busters[buster_id]
            ghost = entities.ghosts[capturing.target_id]
            if distance2(buster.position, ghost.position) > MAX_BUST_DISTANCE ** 2:
                actions[buster_id] = Move(buster_id, ghost.position)   # TODO - anticipate move of ghost?
            else:
                actions[buster_id] = Bust(buster_id, capturing.target_id)

        # Carrying ghost to base
        for buster_id, carrying in self.carrying.items():
            buster = entities.busters[buster_id]
            team_corner = TEAM_CORNERS[entities.my_team]
            if distance2(team_corner, buster.position) < RADIUS_BASE ** 2:
                actions[buster_id] = Release(buster_id, buster.carried_ghost)
            else:
                actions[buster_id] = Move(buster_id, team_corner)
        return actions

    def _state_transitions(self, entities: Entities):
        ghosts: List[Ghost] = list(entities.ghosts.values())

        debug(self.unassigned)
        debug(self.exploring)
        debug(self.capturing)
        debug(self.carrying)

        # Assign vacant busters
        # TODO - avoid assigning too many to the same buster
        vacant_busters = [entities.busters[uid] for uid in self.unassigned]
        assignments = self.territory.assign_destinations(vacant_busters)
        for buster_id in self.unassigned:
            tile_pos = assignments.get(buster_id)
            # TODO - might have no tile_positions
            buster = entities.busters[buster_id]
            best_ghost = min(ghosts, key=lambda g: self._ghost_score(buster, g), default=None)
            if best_ghost and self._ghost_score(buster, best_ghost) <= self._tile_score(buster, tile_pos):
                self.capturing[buster_id] = Capturing(target_id=best_ghost.uid)
            else:
                self.exploring[buster_id] = Exploring(destination=tile_pos)

        self.unassigned.clear()

    def _ghost_score(self, buster: Buster, ghost: Ghost):
        return distance2(ghost.position, buster.position) / MAX_MOVE_DISTANCE ** 2 + ghost.endurance

    def _tile_score(self, buster: Buster, tile_pos: Vector):
        return distance2(tile_pos, buster.position) / MAX_MOVE_DISTANCE ** 2 + 10   # TODO - parameterize (and evolve)


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
    for turn_nb in itertools.count(start=0, step=1):
        entities = read_entities(my_team_id=my_team_id,
                                 busters_per_player=busters_per_player,
                                 ghost_count=ghost_count,
                                 previous_entities=previous_entities)
        debug(entities)

        if turn_nb == 0:
            agent.init(entities)

        for action in agent.get_actions(entities):
            print(action)

        on_end_of_turn(entities)
        previous_entities = entities


if __name__ == '__main__':
    game_loop()
