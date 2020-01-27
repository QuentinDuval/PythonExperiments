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


def rotate(vector: Vector, angle: Angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x = vector[0] * cos_a - vector[1] * sin_a
    y = vector[0] * sin_a + vector[1] * cos_a
    return np.array([x, y])


"""
------------------------------------------------------------------------------------------------------------------------
GAME CONSTANTS
------------------------------------------------------------------------------------------------------------------------
"""

WIDTH = 16000
HEIGHT = 9000

RADIUS_BASE = 1600
RADIUS_SIGHT = 2200
RADIUS_RADAR = 2 * RADIUS_SIGHT

MAX_GHOST_MOVE_DISTANCE = 400
MAX_MOVE_DISTANCE = 800

MIN_BUST_DISTANCE = 900
MAX_BUST_DISTANCE = 1760
MAX_STUN_DISTANCE = 1760

STUN_DURATION = 10
STUN_COOLDOWN = 20

TEAM_CORNERS = np.array([[0, 0], [WIDTH, HEIGHT]], dtype=np.float32)

"""
------------------------------------------------------------------------------------------------------------------------
MAIN DATA STRUCTURE to keep STATE
------------------------------------------------------------------------------------------------------------------------
"""

EntityId = int


class RadarStatus(enum.Enum):
    NO_RADAR = 0
    HAS_RADAR = 1
    USED_RADAR = 2


@dataclass(frozen=False)
class Ghost:
    uid: EntityId
    position: np.ndarray
    endurance: int
    total_busters_count: int = 0
    my_busters_count: int = 0
    last_seen: int = 0

    @property
    def his_buster_count(self):
        return self.total_busters_count - self.my_busters_count

    def clone(self):
        return copy.deepcopy(self)

    def __lt__(self, other):
        return self.uid < other.uid


@dataclass(frozen=False)
class Buster:  # TODO - separate data structure (and list) for OPPONENT BUSTERS (last known position, anticipated, etc.)
    uid: EntityId
    position: np.ndarray
    team: int  # 0 for team 0, 1 for team 1
    carried_ghost: EntityId  # ID of the ghost being carried, -1 if no ghost carried
    busted_ghost: EntityId  # ID of the ghost being busted, -1 if no ghost busted
    last_seen: int = 0  # Mostly useful for opponent
    stun_duration: int = 0  # How many rounds until end of stun
    stun_cooldown: int = 0
    radar_status: RadarStatus = RadarStatus.HAS_RADAR
    sight_radius: float = RADIUS_SIGHT

    @property
    def carrying_ghost(self) -> bool:
        return self.carried_ghost >= 0

    @property
    def busting_ghost(self) -> bool:
        return self.busted_ghost >= 0

    def clone(self):
        return copy.deepcopy(self)

    def __lt__(self, other):
        return self.uid < other.uid


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

    @property
    def my_corner(self) -> Vector:
        return TEAM_CORNERS[self.my_team]

    @property
    def his_corner(self) -> Vector:
        return TEAM_CORNERS[1 - self.my_team]

    def get_player_busters(self) -> Iterator[Buster]:
        return (b for _, b in self.busters.items() if b.team == self.my_team)

    def get_opponent_busters(self) -> Iterator[Buster]:
        return (b for _, b in self.busters.items() if b.team != self.my_team)

    def get_ghosts(self) -> List[Ghost]:
        return list(self.ghosts.values())

    def get_carried_ghost_ids(self) -> Set[EntityId]:
        return {b.carried_ghost for _, b in self.busters.items() if b.carrying_ghost}

    def clone(self):
        return copy.deepcopy(self)


"""
------------------------------------------------------------------------------------------------------------------------
READING & TRACKING GAME STATE
------------------------------------------------------------------------------------------------------------------------
"""


def in_line_of_sight_of(position: Vector, busters: List[Buster]):
    for buster in busters:
        if distance2(buster.position, position) < RADIUS_SIGHT ** 2:
            return True
    return False


def in_stun_range(buster: Buster, entities: Entities):
    if buster.stun_cooldown > 0:
        return None

    candidate = None
    for opponent in entities.get_opponent_busters():
        if opponent.stun_duration <= 1 and distance2(buster.position, opponent.position) < MAX_STUN_DISTANCE ** 2:
            if opponent.carrying_ghost:
                return opponent
            candidate = opponent
    return candidate


def past_ghost_relevant(entities: Entities, ghost: Ghost):
    # TODO - make it an information later in my algorithms
    distance_my_team = distance(ghost.position, entities.my_corner)
    distance_his_team = distance(ghost.position, entities.his_corner)
    threshold = 100 if distance_my_team < distance_his_team * 0.8 else 25  # TODO - take into account opponents
    threshold /= math.sqrt(entities.busters_per_player)
    return ghost.last_seen <= threshold


def complete_picture_from_past(entities: Entities, previous_entities: Entities):
    # Deduce general information
    entities.my_score = previous_entities.my_score
    entities.current_turn = previous_entities.current_turn + 1

    # Complete known buster information
    my_ghost_count = defaultdict(int)
    for buster in entities.get_player_busters():
        previous_buster = previous_entities.busters[buster.uid]
        buster.stun_cooldown = previous_buster.stun_cooldown - 1
        if buster.busting_ghost and previous_buster.busting_ghost:
            buster.busted_ghost = previous_buster.busted_ghost
            my_ghost_count[buster.busted_ghost] += 1
        if previous_buster.radar_status == RadarStatus.USED_RADAR:
            buster.sight_radius = RADIUS_RADAR
            buster.radar_status = RadarStatus.NO_RADAR
        else:
            buster.radar_status = previous_buster.radar_status

    # Complete known ghost information
    for ghost in entities.get_ghosts():
        if ghost.total_busters_count > 0:
            ghost.my_busters_count = my_ghost_count[ghost.uid]

    # Adding missing ghosts
    carried_ghosts = entities.get_carried_ghost_ids()
    my_busters = list(entities.get_player_busters())
    for ghost_id, ghost in previous_entities.ghosts.items():
        if ghost_id not in entities.ghosts and ghost_id not in carried_ghosts:
            if not in_line_of_sight_of(ghost.position, my_busters):
                if past_ghost_relevant(entities, ghost):
                    entities.ghosts[ghost_id] = ghost
                    ghost.last_seen += 1
                    # TODO - make ghosts move away from their closest buster

    # Adding missing opponent busters
    for buster in previous_entities.get_opponent_busters():
        if buster.uid not in entities.busters and not in_line_of_sight_of(buster.position, my_busters):
            opponent_corner = entities.his_corner

            # Moving them to their base if carrying
            if buster.carrying_ghost and distance2(opponent_corner, buster.position) > RADIUS_BASE ** 2:
                direction = opponent_corner - buster.position
                buster.position += direction / norm(direction) * MAX_MOVE_DISTANCE
                entities.busters[buster.uid] = buster
                buster.last_seen = 1

            # Else keeping a timer, but decreasing the belief
            elif buster.last_seen <= 20: # TODO - increase credibility based on busting or not
                entities.busters[buster.uid] = buster
                buster.carried_ghost = -1
                buster.last_seen += 1


def read_entities(my_team_id: int, busters_per_player: int, ghost_count: int, previous_entities: Entities):
    entities = Entities(my_team=my_team_id, my_score=0,
                        busters_per_player=busters_per_player,
                        ghost_count=ghost_count)

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
                total_busters_count=value)
        else:
            entities.busters[identity] = Buster(
                uid=identity,
                position=position,
                team=entity_type,
                carried_ghost=value if state == 1 else -1,
                stun_duration=value if state == 2 else 0,
                busted_ghost=0 if state == 3 else -1)

    # IMPORTANT NODE:
    # * the carried ghosts are not part of the input

    if previous_entities:
        complete_picture_from_past(entities, previous_entities)
    return entities


"""
------------------------------------------------------------------------------------------------------------------------
ACTION
------------------------------------------------------------------------------------------------------------------------
"""


class Move(NamedTuple):
    caster_id: int
    position: Union[Tuple[float, float], Vector]

    def __repr__(self):
        x, y = self.position
        return "MOVE " + str(int(x)) + " " + str(int(y))


class Bust:
    def __init__(self, entities: Entities, caster_id: int, target_id: int):
        self.caster_id = caster_id
        self.target_id = target_id
        entities.busters[caster_id].busted_ghost = target_id

    def __repr__(self):
        return "BUST " + str(self.target_id)


class Release:
    def __init__(self, entities: Entities, caster_id: int, target_id: int):
        self.caster_id = caster_id
        self.target_id = target_id
        entities.busters[self.caster_id].carried_ghost = -1

    def __repr__(self):
        return "RELEASE"


class Stun:
    def __init__(self, entities: Entities, caster_id: int, target_id: int):
        self.caster_id = caster_id
        self.target_id = target_id
        entities.busters[self.caster_id].stun_cooldown = STUN_COOLDOWN
        entities.busters[self.target_id].stun_duration = STUN_DURATION

    def __repr__(self):
        return "STUN " + str(self.target_id)


class Radar:
    def __init__(self, entities: Entities, caster_id: int):
        entities.busters[caster_id].radar_status = RadarStatus.USED_RADAR

    def __repr__(self):
        return "RADAR"


class Eject:
    def __init__(self, entities: Entities, target_id: int, destination: Vector):
        self.destination = destination
        entities.busters[target_id].carried_ghost = -1
        # TODO - does the creation of the ghost happen in the same turn? if so, make it appear

    def __repr__(self):
        x, y = self.destination
        return "EJECT " + str(int(x)) + " " + str(int(y))


Action = Union[Move, Bust, Release, Stun, Radar, Eject]

"""
------------------------------------------------------------------------------------------------------------------------
TERRITORY - TO GUIDE EXPLORATION
------------------------------------------------------------------------------------------------------------------------
"""


class Territory:
    def __init__(self, team_id: int, w=15, h=10):
        self.unvisited: MutableSet[Tuple[int, int]] = set()
        self.w = w
        self.h = h
        self.cell_width = WIDTH / self.w
        self.cell_height = HEIGHT / self.h
        self.cells = np.zeros(shape=(self.w, self.h, 2))
        for i in range(self.w):
            for j in range(self.h):
                self.cells[i, j, 0] = self.cell_width / 2 + self.cell_width * i
                self.cells[i, j, 1] = self.cell_height / 2 + self.cell_height * j
                # No reason to explore opponent's base
                if distance2(self.cells[i, j], TEAM_CORNERS[1 - team_id]) > RADIUS_BASE ** 2:
                    self.unvisited.add((i, j))

        center = (self.w / 2, self.h / 2)
        self.heat = np.ones(shape=(self.w, self.h), dtype=np.float32)
        for i in range(self.w):
            for j in range(self.h):
                self.heat[(i, j)] = 10 / (1 + distance((i, j), center))

    def __len__(self):
        return len(self.unvisited)

    def assign_destinations(self, busters: Collection[Buster]) -> Dict[int, Tuple[float, float]]:
        heap = []
        for buster in busters:
            for i, j in self.unvisited:
                point = self.cells[i, j]
                heat = self._weight_of_cell(i, j)
                dist2 = distance2(point, buster.position)
                heapq.heappush(heap, (dist2 / heat, buster.uid, tuple(point)))

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
        for buster in busters:
            for i, j in list(self.unvisited):
                point = self.cells[i, j]
                if all(distance2(corner, buster.position) < buster.sight_radius ** 2 for corner in
                       self._corners_of(point)):
                    self.unvisited.discard((i, j))

    def is_visited(self, point: Tuple[float, float]) -> bool:
        i = int(point[0] / self.cell_width)
        j = int(point[1] / self.cell_height)
        return (i, j) not in self.unvisited

    def _corners_of(self, position: Tuple[float, float]):
        for dx in (-1, 1):
            for dy in (-1, 1):
                yield position[0] + dx * self.cell_width, position[1] + dy * self.cell_height

    def _weight_of_cell(self, i: int, j: int):
        # Prioritize strategic cells + with a lot of unexplored neighbors
        heat = self.heat[i, j]
        for di in (-1, 1):
            for dj in (-1, 1):
                if (i + di, j + dj) in self.unvisited:
                    heat += self.heat[i + di, j + dj] * 0.5
        return heat


"""
------------------------------------------------------------------------------------------------------------------------
AGENT
------------------------------------------------------------------------------------------------------------------------
"""


@dataclass(frozen=False)
class Opening:
    destination: Tuple[float, float]
    use_radar: bool = False


@dataclass()
class Exploring:
    destination: Tuple[float, float]


@dataclass()
class Capturing:
    target_id: EntityId


@dataclass()
class Carrying:
    pass


@dataclass()
class Escorting:
    target_id: EntityId


@dataclass()
class Intercepting:
    target_id: EntityId


@dataclass()
class Ambushing:
    destination: Vector


'''
@dataclass(frozen=False)
class Herding:
    pass
'''


class Agent:
    # TODO - IDEAS:
    #   - consider going for a stun if the opponent carries a ghost & pursing him
    #   - stick around opponent busters to steal their ghosts
    #   - when you find a ghost for the first time (requires tracking): add a "point of interest" to map
    #   - herding when carrying a ghost back home (and if no danger)
    #   - exit some states with simple exponential decay
    #   - keep RADAR for avoiding ambush when bringing a ghost back to the base
    #   - FIND USE FOR EJECT
    #   - Help busters that requires help nearby... interrupt exploration actions when doing so

    # TODO - FIX:
    #   - do not stun when the opponent is in his base
    #   - when BUSTING, stun opponent only if you can finish the busting before end of stun
    #   - do not bust ghost if more opponent on it, it only helps him
    #   - recompute the destinations based on all busters that are exploring (else same destination possible)

    def __init__(self, team_id: int):
        self.chronometer = Chronometer()
        self.territory = Territory(team_id)
        self.unassigned: MutableSet[int] = set()
        self.opening: Dict[int, Opening] = {}
        self.exploring: Dict[int, Exploring] = {}
        self.capturing: Dict[int, Capturing] = {}
        self.carrying: Dict[int, Carrying] = {}
        self.escorting: Dict[int, Escorting] = {}
        self.intercepting: Dict[int, Intercepting] = {}
        self.ambushing: Dict[int, Ambushing] = {}

    def get_actions(self, entities: Entities) -> List[Action]:
        self.chronometer.start()

        if entities.current_turn == 0:
            self._opening_book(entities)
        else:
            self._update_past_states(entities)  # Exit state which goal is satisfied
            self._strategic_analysis(entities)  # Identifying overall / coordinated goals
            self._assign_vacant_busters(entities)  # Assign busters to their new states

        self.debug_states()
        actions = self._carry_actions(entities)

        debug("Time spent:", self.chronometer.spent(), "ms")
        return [action for buster_id, action in sorted(actions.items(), key=lambda p: p[0])]

    def debug_states(self):
        debug("opening", self.opening)
        debug("exploring", self.exploring)
        debug("capturing", self.capturing)
        debug("carrying", self.carrying)
        debug("escorting", self.escorting)
        debug("intercepting", self.intercepting)
        debug("ambushing", self.ambushing)
        debug("unassigned", self.unassigned)

    """
    --------------------------------------------------------------------------------------------------------------------
    OPENING BOOK: initial strategy
    --------------------------------------------------------------------------------------------------------------------
    """

    def _opening_book(self, entities: Entities):
        # Go to fixed locations, then use the radar at the end

        # Compute the way points
        n = entities.busters_per_player
        angle_diff = math.pi / 2 / (n + 1)
        angle_delta = angle_diff / 2
        angles = [angle_delta + angle_diff * i for i in range(n)]
        if entities.my_team == 0:
            direction = np.array([8800, 0])
        else:
            direction = np.array([-8800, 0])
        team_corner = TEAM_CORNERS[entities.my_team]
        waypoints = [team_corner + rotate(direction, a) for a in angles]

        # Assign the way points to each ghosts
        indices = set(range(n))
        for buster in entities.get_player_busters():
            min_i = min(indices, key=lambda i: distance2(waypoints[i], buster.position))
            self.opening[buster.uid] = Opening(tuple(waypoints[min_i]), use_radar=False)
            indices.remove(min_i)

    """
    --------------------------------------------------------------------------------------------------------------------
    TRACKING CHANGE: existing state
    --------------------------------------------------------------------------------------------------------------------
    """

    def _update_past_states(self, entities: Entities):

        # Tracking opening state
        for buster_id, opening in list(self.opening.items()):
            buster = entities.busters[buster_id]
            if distance2(buster.position, opening.destination) < MAX_MOVE_DISTANCE ** 2:
                if opening.use_radar:
                    del self.opening[buster_id]
                    self.unassigned.add(buster_id)
                else:
                    opening.use_radar = True

        # Dropping exploration state (assignment will recover it if necessary)
        self.territory.track_explored(entities.get_player_busters())
        for buster_id, exploring in self.exploring.items():
            self.unassigned.add(buster_id)
        self.exploring.clear()

        # Carrying / escorting update base on the success of the mission
        for buster_id, carrying in list(self.carrying.items()):

            # If carrying, keep on carrying the ghost toward base
            if entities.busters[buster_id].carrying_ghost:
                continue

            # If not carrying anymore (could be due to STUN or RELEASE or EJECT)
            del self.carrying[buster_id]
            self.unassigned.add(buster_id)
            for escorting_buster_id, escorting in list(self.escorting.items()):
                if escorting.target_id == buster_id:
                    del self.escorting[escorting_buster_id]
                    self.unassigned.add(escorting_buster_id)

        # Capturing update
        for buster_id, capturing in list(self.capturing.items()):
            buster = entities.busters[buster_id]

            # If successful capture of the ghost
            if buster.carrying_ghost:
                del self.capturing[buster_id]
                self.carrying[buster_id] = Carrying()

            # In case of stun / end of capture by another buster
            elif not buster.busting_ghost:
                del self.capturing[buster_id]
                self.unassigned.add(buster_id)

        # Intercepting update
        for buster_id, intercepting in list(self.intercepting.items()):
            opponent_buster = entities.busters.get(intercepting.target_id)
            if not opponent_buster or not opponent_buster.carrying_ghost:
                del self.intercepting[buster_id]
                self.unassigned.add(buster_id)

        # Ambushing update
        threshold_dist2 = 3 * RADIUS_BASE ** 2
        released_ghosts = [g.uid for g in entities.get_ghosts()
                           if g.endurance == 0 and distance2(g.position, entities.his_corner) < threshold_dist2]
        for buster_id, ambushing in list(self.ambushing.items()):
            if released_ghosts:
                buster = entities.busters[buster_id]
                if buster.stun_cooldown >= STUN_COOLDOWN / 2:
                    del self.ambushing[buster_id]
                    self.capturing[buster_id] = Capturing(released_ghosts.pop())

    """
    --------------------------------------------------------------------------------------------------------------------
    STRATEGIC ANALYSIS: identifying coordinated goals
    --------------------------------------------------------------------------------------------------------------------
    """

    def _strategic_analysis(self, entities: Entities):

        # Abandon a fight you cannot win, and plan ambush - TODO: avoid the fight, ban the ghost, look from distance?
        for buster_id, capturing in list(self.capturing.items()):
            ghost_id = capturing.target_id
            ghost = entities.ghosts[ghost_id]
            if ghost.his_buster_count > ghost.my_busters_count:
                del self.capturing[buster_id]
                self.ambushing[buster_id] = Ambushing(entities.his_corner)

        # Mandate an interception
        for opponent in entities.get_opponent_busters():
            if opponent.carrying_ghost:
                buster_id = self._assign_interceptor(entities, opponent)
                if buster_id is not None:
                    self.intercepting[buster_id] = Intercepting(opponent.uid)

        # TODO - identify when we should DROP an activity (identify a top of activities)
        # TODO - identify whether or not we should drop a ghost capture (bad situation)
        # TODO - identify whether or not we should attack an opponent position
        # TODO - overall, should give cost to ghosts... or COORDINATED METRICS / GHOSTS at least

    def _assign_interceptor(self, entities: Entities, opponent: Buster) -> Optional[EntityId]:
        # TODO - try the closest? try the one with the least useful thing to do
        for buster_id in list(self.unassigned):
            if self._can_intercept(entities, entities.busters[buster_id], opponent):
                self.unassigned.remove(buster_id)
                return buster_id

        for buster_id in list(self.capturing):
            if self._can_intercept(entities, entities.busters[buster_id], opponent):
                del self.capturing[buster_id]
                return buster_id

        for buster_id in list(self.ambushing):
            if self._can_intercept(entities, entities.busters[buster_id], opponent):
                del self.ambushing[buster_id]
                return buster_id

    def _can_intercept(self, entities: Entities, buster: Buster, opponent: Buster):
        my_dist = distance(buster.position, entities.his_corner) - RADIUS_BASE
        his_dist = distance(opponent.position, entities.his_corner) - RADIUS_BASE
        return my_dist < his_dist and buster.stun_cooldown < his_dist / MAX_MOVE_DISTANCE

    """
    --------------------------------------------------------------------------------------------------------------------
    STATE TRANSITIONS: entering new states
    --------------------------------------------------------------------------------------------------------------------
    """

    def _assign_vacant_busters(self, entities: Entities):
        ghosts: List[Ghost] = list(entities.ghosts.values())
        busters = [entities.busters[uid] for uid in self.unassigned]
        destinations = self.territory.assign_destinations(busters)

        # Identify the ghosts that are already taken
        ghosts_taken_count = defaultdict(int)
        for ghost in entities.get_ghosts():
            ghosts_taken_count[ghost.uid] = ghost.my_busters_count

        # Consider all possible actions and their priority - TODO generalize to: escorting, intercepting...
        heap: List[Tuple[float, Buster, Ghost, int]] = []
        for ghost in ghosts:
            for buster in busters:
                busting_count = ghosts_taken_count[ghost.uid]
                heapq.heappush(heap, (
                self._ghost_score(entities, buster, ghost, busting_count), buster, ghost, busting_count))

        # A kind of Dijkstra for assigning ghost to their best priority
        while self.unassigned and heap:
            ghost_score, buster, ghost, busting_count = heapq.heappop(heap)
            if buster.uid not in self.unassigned:
                continue

            if ghosts_taken_count[ghost.uid] > busting_count:
                busting_count = ghosts_taken_count[ghost.uid]
                heapq.heappush(heap, (
                self._ghost_score(entities, buster, ghost, busting_count), buster, ghost, busting_count))
                continue

            tile_pos = destinations.get(buster.uid)
            if ghost_score < self._tile_score(buster, tile_pos):
                self.capturing[buster.uid] = Capturing(target_id=ghost.uid)
                ghosts_taken_count[ghost.uid] += 1
            else:
                self.exploring[buster.uid] = Exploring(destination=tile_pos)
            self.unassigned.remove(buster.uid)

        # Go for tiles (in case the heap of ghost is empty)
        for buster_id in list(self.unassigned):
            tile_pos = destinations.get(buster_id)
            if tile_pos:
                self.exploring[buster_id] = Exploring(destination=tile_pos)
                self.unassigned.remove(buster_id)

        # No ghosts / no exploration left: go for escort / attack
        carrying_allies = [b for b in entities.get_player_busters() if b.carrying_ghost]
        for buster_id in self.unassigned:
            buster = entities.busters[buster_id]
            closest_ally = min(carrying_allies, key=lambda b: distance2(b.position, buster.position), default=None)
            if closest_ally and np.random.rand(1) < 0.5:  # TODO - use more of this (weight it though)
                self.escorting[buster.uid] = Escorting(closest_ally.uid)
            else:
                self.ambushing[buster.uid] = Ambushing(destination=self._ambushing_pos(entities))
        self.unassigned.clear()

    def _ghost_score(self, entities: Entities, buster: Buster, ghost: Ghost, busting_count: int):
        steps_to_target = distance2(ghost.position, buster.position) / MAX_MOVE_DISTANCE ** 2
        ghost_value = 10 + (entities.current_turn + 1) / 10 - ghost.last_seen / 10

        # Take into account:
        # - the cost of resources by diminishing the value of the ghost
        # - the duration till we get the point

        # TODO - should take into account:
        #   - the balance with opponent busters
        #   - the fact that we do not need too many busters on the ghost
        #   => MOVE THIS STRATEGIC OVERVIEW, TOO HARD HERE

        endurance_at_arrival = ghost.endurance - steps_to_target * ghost.total_busters_count
        ghost_value /= (busting_count + 1) ** 2
        return (steps_to_target + endurance_at_arrival / (busting_count + 1)) / ghost_value

    def _tile_score(self, buster: Buster, tile_pos: Tuple[float, float]):
        if not tile_pos:
            return float('inf')

        nb_steps = distance2(tile_pos, buster.position) / MAX_MOVE_DISTANCE ** 2
        exploration_value = 1  # TODO - depend on where we are in the game
        return nb_steps / exploration_value

    """
    --------------------------------------------------------------------------------------------------------------------
    HANDLING ACTIONS: carrying out the actions dictated by states
    --------------------------------------------------------------------------------------------------------------------
    """

    def _carry_actions(self, entities: Entities) -> Dict[int, Action]:
        actions: Dict[EntityId, Action] = dict()
        self._carry_opening(entities, actions)
        self._carry_map_exploration(entities, actions)
        self._carry_ghost_capture(entities, actions)
        self._carry_ghost_to_base(entities, actions)
        self._carry_intercepting(entities, actions)
        self._carry_ambushing(entities, actions)
        return actions

    def _carry_opening(self, entities: Entities, actions: Dict[EntityId, Action]):
        for buster_id, opening in self.opening.items():
            if opening.use_radar:
                actions[buster_id] = Radar(entities, buster_id)
            else:
                actions[buster_id] = Move(buster_id, opening.destination)

    def _carry_map_exploration(self, entities: Entities, actions: Dict[EntityId, Action]):
        for buster_id, exploring in self.exploring.items():
            buster = entities.busters[buster_id]
            opponent = in_stun_range(buster, entities)
            if opponent:
                actions[buster_id] = Stun(entities, buster_id, opponent.uid)
            else:
                actions[buster_id] = Move(buster_id, exploring.destination)

    def _carry_ghost_capture(self, entities: Entities, actions: Dict[EntityId, Action]):
        # TODO - improve in highly congested area - how to do GOOD STUNS
        for buster_id, capturing in self.capturing.items():
            buster = entities.busters[buster_id]
            ghost = entities.ghosts[capturing.target_id]
            dist2 = distance2(buster.position, ghost.position)
            opponent = in_stun_range(buster, entities)
            if opponent:
                actions[buster_id] = Stun(entities, buster_id, opponent.uid)
            elif dist2 > MAX_BUST_DISTANCE ** 2:
                actions[buster_id] = Move(buster_id, ghost.position)
            elif dist2 == 0:
                actions[buster_id] = Move(buster_id, np.array([WIDTH / 2, HEIGHT / 2]))
            elif dist2 < MIN_BUST_DISTANCE ** 2:
                direction = buster.position - ghost.position
                destination = buster.position + direction / norm(direction) * math.sqrt(MIN_BUST_DISTANCE ** 2 - dist2)
                actions[buster_id] = Move(buster_id, destination)
            else:
                actions[buster_id] = Bust(entities, caster_id=buster_id, target_id=capturing.target_id)

    def _carry_ghost_to_base(self, entities: Entities, actions: Dict[EntityId, Action]):
        # Carrying ghost to base
        for buster_id, carrying in self.carrying.items():
            buster = entities.busters[buster_id]
            team_corner = entities.my_corner
            opponent = in_stun_range(entities.busters[buster_id], entities)
            if opponent and opponent.stun_cooldown <= 0:
                actions[buster_id] = Stun(entities, buster_id, opponent.uid)
            elif distance2(team_corner, buster.position) < RADIUS_BASE ** 2:
                actions[buster_id] = Release(entities, buster_id, buster.carried_ghost)
            else:
                actions[buster_id] = Move(buster_id, team_corner)

        # Assisting an ally
        for buster_id, escorting in self.escorting.items():
            target_pos = entities.busters[escorting.target_id].position
            opponent = in_stun_range(entities.busters[buster_id], entities)
            if opponent:
                actions[buster_id] = Stun(entities, buster_id, opponent.uid)
            else:
                actions[buster_id] = Move(buster_id, target_pos * 0.5 + entities.my_corner * 0.5)

    def _carry_intercepting(self, entities: Entities, actions: Dict[EntityId, Action]):
        # TODO - use EJECT somewhere here
        for buster_id, intercepting in list(self.intercepting.items()):
            buster = entities.busters[buster_id]
            opponent = entities.busters[intercepting.target_id]

            # Reached the opponent, stun him
            if distance2(opponent.position, buster.position) < MAX_STUN_DISTANCE ** 2 and buster.stun_cooldown <= 0:
                actions[buster_id] = Stun(entities, buster_id, intercepting.target_id)
                del self.intercepting[buster_id]
                self.unassigned.add(buster_id)
                continue

            # Seeking the opponent
            direction = entities.his_corner - opponent.position
            remaining_distance = norm(direction)
            direction /= remaining_distance
            normal = np.array([-direction[1], direction[0]])
            dist_to_trajectory = abs(np.dot(normal, buster.position - opponent.position))
            if dist_to_trajectory < MAX_STUN_DISTANCE:
                actions[buster_id] = Move(buster_id, opponent.position)
            else:
                direction *= 0.7 * (remaining_distance - RADIUS_BASE)
                actions[buster_id] = Move(buster_id, opponent.position + direction)

    def _carry_ambushing(self, entities: Entities, actions: Dict[EntityId, Action]):
        # TODO - use EJECT somewhere here
        intercept_dir = self._ambushing_dir(entities)
        for buster_id, ambushing in self.ambushing.items():
            buster = entities.busters[buster_id]
            opponent = in_stun_range(entities.busters[buster_id], entities)  # TODO - do better here - pursue
            if opponent:
                actions[buster_id] = Stun(entities, buster_id, opponent.uid)
            elif distance2(buster.position, ambushing.destination) < 16:
                angle = np.random.choice([-math.pi / 10, 0, math.pi / 10, math.pi / 5])
                destination = entities.his_corner + rotate(intercept_dir, angle)
                actions[buster_id] = Move(buster_id, destination)
                self.ambushing[buster_id] = Ambushing(destination)  # TODO - move to assignment of orders?
            else:
                actions[buster_id] = Move(buster_id, ambushing.destination)

    def _ambushing_pos(self, entities: Entities):
        intercept_dir = self._ambushing_dir(entities)
        return entities.his_corner + intercept_dir

    def _ambushing_dir(self, entities: Entities):
        intercept_dir = entities.my_corner - entities.his_corner
        return intercept_dir / norm(intercept_dir) * RADIUS_BASE * 2


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

    agent = Agent(my_team_id)
    previous_entities = None
    while True:
        entities = read_entities(my_team_id=my_team_id,
                                 busters_per_player=busters_per_player,
                                 ghost_count=ghost_count,
                                 previous_entities=previous_entities)

        # TODO - shorten this display to make it useful
        # debug(entities)

        for action in agent.get_actions(entities):
            print(action)

        previous_entities = entities


if __name__ == '__main__':
    game_loop()
