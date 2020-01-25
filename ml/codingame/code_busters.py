import abc
import copy
import enum
from collections import *
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
    current_turn: int = 0

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
    entities = Entities(my_team=my_team_id, my_score=0,
                        busters_per_player=busters_per_player,
                        ghost_count=ghost_count)

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
        position = np.array([x, y], dtype=np.float64)
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
    entities.current_turn = previous_entities.current_turn + 1
    for buster in entities.get_player_busters():
        buster.stun_cooldown = previous_entities.busters[buster.uid].stun_cooldown - 1

    # Adding missing ghosts
    for ghost_id, ghost in previous_entities.ghosts.items():
        if ghost_id not in entities.ghosts and ghost_id not in carried_ghosts and ghost.last_seen <= 10:
            entities.ghosts[ghost_id] = ghost
            ghost.last_seen += 1
            # TODO - make ghosts move away from their closest buster

    # Adding missing opponent busters
    for buster in previous_entities.get_opponent_busters():
        if buster.uid not in entities.busters:
            opponent_corner = TEAM_CORNERS[1-entities.my_team]

            # Moving them to their base if carrying
            if buster.carried_ghost >= 0 and distance2(opponent_corner, buster.position) > RADIUS_BASE ** 2:
                direction = TEAM_CORNERS[1-entities.my_team] - buster.position
                buster.position += direction / norm(direction) * MAX_MOVE_DISTANCE
                entities.busters[buster.uid] = buster
                buster.last_seen = 1

            # Else keeping a timer, but decreasing the belief
            if buster.last_seen <= 10:
                entities.busters[buster.uid] = buster
                buster.last_seen += 1
    return entities


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
    target_id: EntityId


@dataclass(frozen=False)
class Intercepting:
    pass


@dataclass(frozen=False)
class Herding:
    pass


class Agent:
    def __init__(self):
        self.chrono = Chronometer()
        self.territory = Territory()
        self.unassigned: Set[int] = set()
        self.exploring: Dict[int, Exploring] = {}
        self.capturing: Dict[int, Capturing] = {}
        self.carrying: Dict[int, Carrying] = {}
        self.escorting: Dict[int, Escorting] = {}
        self.intercepting: Dict[int, Intercepting] = {}

    def init(self, entities: Entities):
        busters = list(entities.get_player_busters())
        assignments = self.territory.assign_destinations(busters)
        for buster in entities.get_player_busters():
            self.exploring[buster.uid] = Exploring(assignments[buster.uid])

    def get_actions(self, entities: Entities) -> List[Action]:
        self.chrono.start()

        self._update_past_states(entities)
        self._assign_vacant_busters(entities)
        # self._strategic_analysis(entities) # TODO - set the goals ?
        actions = self._carry_actions(entities)

        debug("Time spent:", self.chrono.spent())
        return [p[1] for p in sorted(actions.items(), key=lambda p: p[0])]

    """
    --------------------------------------------------------------------------------------------------------------------
    TRACKING CHANGE
    --------------------------------------------------------------------------------------------------------------------
    """

    def _update_past_states(self, entities: Entities):

        # Tracking exploration state
        self.territory.track_explored(entities.get_player_busters())
        for buster_id, exploring in list(self.exploring.items()):
            if self.territory.is_visited(exploring.destination):
                del self.exploring[buster_id]
                self.unassigned.add(buster_id)

        # Carrying / escorting update
        for buster_id, carrying in list(self.carrying.items()):
            if entities.busters[buster_id].carried_ghost < 0:
                del self.carrying[buster_id]
                self.unassigned.add(buster_id)
                for escorting_buster_id, escorting in list(self.escorting.items()):
                    if escorting.target_id == buster_id:
                        del self.escorting[escorting_buster_id]
                        self.unassigned.add(escorting_buster_id)

        # Capturing update
        for buster_id, capturing in list(self.capturing.items()):
            if entities.busters[buster_id].carried_ghost >= 0:
                del self.capturing[buster_id]
                self.carrying[buster_id] = Carrying()
            elif not entities.busters[buster_id].busting_ghost:
                del self.capturing[buster_id]
                self.unassigned.add(buster_id)

        # Intercepting update
        # TODO - maybe check the success rate ? simple exponential decay? out of stun?

    """
    --------------------------------------------------------------------------------------------------------------------
    STATE TRANSITIONS
    --------------------------------------------------------------------------------------------------------------------
    """

    def _assign_vacant_busters(self, entities: Entities):
        ghosts: List[Ghost] = list(entities.ghosts.values())
        busters = [entities.busters[uid] for uid in self.unassigned]
        destinations = self.territory.assign_destinations(busters)

        heap: List[Tuple[float, Buster, Ghost, int]] = []
        for ghost in ghosts:
            for buster in busters:
                heapq.heappush(heap, (self._ghost_score(buster, ghost, 0), buster, ghost, 0))

        # TODO - count the busters per ghosts ALREADY THERE and fight for equality (or remove your busters if not worth)
        ghosts_taken_count = defaultdict(int)
        while self.unassigned and heap:
            ghost_score, buster, ghost, busting_count = heapq.heappop(heap)
            if buster.uid not in self.unassigned:
                continue

            if ghosts_taken_count[ghost.uid] > busting_count:
                busting_count = ghosts_taken_count[ghost.uid]
                heapq.heappush(heap, (self._ghost_score(buster, ghost, busting_count), buster, ghost, busting_count))
                continue

            tile_pos = destinations.get(buster.uid)
            if ghost_score < self._tile_score(buster, tile_pos):
                self.capturing[buster.uid] = Capturing(target_id=ghost.uid)
                ghosts_taken_count[ghost.uid] += 1
            else:
                self.exploring[buster.uid] = Exploring(destination=tile_pos)
            self.unassigned.remove(buster.uid)

        # No ghosts left: go for escort / attack
        carrying_allies = [b for b in entities.get_player_busters() if b.carried_ghost >= 0]
        for buster_id in self.unassigned:
            buster = entities.busters[buster_id]
            closest_ally = min(carrying_allies, key=lambda b: distance2(b.position, buster.position), default=None)
            if closest_ally and np.random.rand(1) < 0.5: # TODO - use more of this (weight it though)
                self.escorting[buster.uid] = Escorting(closest_ally.uid)
            else:
                self.intercepting[buster.uid] = Intercepting()
        self.unassigned.clear()

    def _ghost_score(self, buster: Buster, ghost: Ghost, busting_count: int):
        nb_steps = distance2(ghost.position, buster.position) / MAX_MOVE_DISTANCE ** 2
        ghost_value = 10        # TODO - depend on where we are in the game
        busting_count += 1      # The number of busters once we take this one
        return busting_count * (nb_steps + ghost.endurance / busting_count) / ghost_value

    def _tile_score(self, buster: Buster, tile_pos: Vector):
        if not tile_pos:
            return float('inf')

        nb_steps = distance2(tile_pos, buster.position) / MAX_MOVE_DISTANCE ** 2
        exploration_value = 1   # TODO - depend on where we are in the game
        return nb_steps / exploration_value

    """
    --------------------------------------------------------------------------------------------------------------------
    HANDLING ACTIONS
    --------------------------------------------------------------------------------------------------------------------
    """

    def _carry_actions(self, entities: Entities) -> Dict[int, Action]:
        actions: Dict[EntityId, Action] = dict()
        self._carry_map_exploration(entities, actions)
        self._carry_ghost_capture(entities, actions)
        self._carry_ghost_to_base(entities, actions)
        self._carry_interception(entities, actions)
        return actions

    def _carry_map_exploration(self, entities: Entities, actions: Dict[EntityId, Action]):
        for buster_id, exploring in self.exploring.items():
            actions[buster_id] = Move(buster_id, exploring.destination)

    def _carry_ghost_capture(self, entities: Entities, actions: Dict[EntityId, Action]):
        for buster_id, capturing in self.capturing.items():
            buster = entities.busters[buster_id]
            ghost = entities.ghosts[capturing.target_id]
            dist2 = distance2(buster.position, ghost.position)
            if dist2 > MAX_BUST_DISTANCE ** 2:
                actions[buster_id] = Move(buster_id, ghost.position)   # TODO - anticipate move of ghost?
            elif dist2 == 0:
                actions[buster_id] = Move(buster_id, np.array([WIDTH/2, HEIGHT/2]))
            elif dist2 < MIN_BUST_DISTANCE ** 2:
                direction = buster.position - ghost.position
                destination = buster.position + direction / norm(direction) * math.sqrt(MIN_BUST_DISTANCE ** 2 - dist2)
                actions[buster_id] = Move(buster_id, destination)
            else:
                actions[buster_id] = Bust(buster_id, capturing.target_id)

    def _carry_ghost_to_base(self, entities: Entities, actions: Dict[EntityId, Action]):
        # Carrying ghost to base
        for buster_id, carrying in self.carrying.items():
            buster = entities.busters[buster_id]
            team_corner = TEAM_CORNERS[entities.my_team]
            if distance2(team_corner, buster.position) < RADIUS_BASE ** 2:
                actions[buster_id] = Release(buster_id, buster.carried_ghost)
            else:
                actions[buster_id] = Move(buster_id, team_corner)

        # Assisting an ally
        for buster_id, escorting in self.escorting.items():
            target_pos = entities.busters[escorting.target_id].position
            team_corner = TEAM_CORNERS[entities.my_team]
            actions[buster_id] = Move(buster_id, target_pos * 0.5 + team_corner * 0.5)

    def _carry_interception(self, entities: Entities, actions: Dict[EntityId, Action]):
        for buster_id, intercepting in self.intercepting.items():
            intercept_pos = TEAM_CORNERS[entities.my_team] * 0.2 + TEAM_CORNERS[1-entities.my_team] * 0.8
            actions[buster_id] = Move(buster_id, intercept_pos)


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

        previous_entities = entities


if __name__ == '__main__':
    game_loop()
