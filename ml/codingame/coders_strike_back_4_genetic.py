from dataclasses import *
import math
import sys
import itertools
from typing import *
import time

import numpy as np

"""
------------------------------------------------------------------------------------------------------------------------
GAME CONSTANTS
------------------------------------------------------------------------------------------------------------------------
"""

WIDTH = 16000
HEIGHT = 9000

TOP_LEFT = (0, 0)
BOT_RIGHT = (WIDTH - 1, HEIGHT - 1)

CHECKPOINT_RADIUS = 600
FORCE_FIELD_RADIUS = 400
VEHICLE_MASS = 1.0

MAX_TURN_DEG = 18
MAX_TURN_RAD = MAX_TURN_DEG / 360 * 2 * math.pi

THRUST_STRENGTH = 200
BOOST_STRENGTH = 650

FIRST_RESPONSE_TIME = 1000
RESPONSE_TIME = 75

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
DATA STRUCTURES
------------------------------------------------------------------------------------------------------------------------
"""

Checkpoint = np.ndarray
CheckpointId = int
Thrust = int


def turn_angle(prev_angle: Angle, diff_angle: Angle) -> Angle:
    if diff_angle > 0:
        diff_angle = min(MAX_TURN_RAD, diff_angle)
    elif diff_angle < 0:
        diff_angle = max(-MAX_TURN_RAD, diff_angle)
    return mod_angle(prev_angle + diff_angle)


@dataclass(frozen=False)
class Entities:
    positions: np.ndarray
    speeds: np.ndarray
    directions: np.ndarray
    masses: np.ndarray
    next_checkpoint_id: np.ndarray
    current_lap: np.ndarray
    boost_available: np.ndarray

    @classmethod
    def empty(cls, size: int):
        return Entities(
            positions=np.zeros(shape=(size, 2)),
            speeds=np.zeros(shape=(size, 2)),
            directions=np.zeros(shape=size),
            masses=np.zeros(shape=size),
            next_checkpoint_id=np.zeros(shape=size, dtype=np.int64),
            current_lap=np.zeros(shape=size, dtype=np.int64),
            boost_available=np.zeros(shape=size, dtype=bool)
        )

    def __len__(self):
        return self.positions.shape[0]

    def __eq__(self, other):
        return np.array_equal(self.positions, other.positions) \
               and np.array_equal(self.speeds, other.speeds) \
               and np.array_equal(self.directions, other.directions) \
               and np.array_equal(self.next_checkpoint_id, other.next_checkpoint_id) \
               and np.array_equal(self.current_lap, other.current_lap)

    def clone(self):
        return Entities(
            positions=self.positions.copy(),
            speeds=self.speeds.copy(),
            directions=self.directions.copy(),
            masses=self.masses.copy(),
            next_checkpoint_id=self.next_checkpoint_id.copy(),
            current_lap=self.current_lap.copy(),
            boost_available=self.boost_available.copy())

    def __repr__(self):
        return "positions:\n" + repr(self.positions) + "\nspeeds:\n" + repr(self.speeds) + "\n"


# TODO - depreciate this
class Vehicle(NamedTuple):
    position: Vector
    speed: Vector
    direction: Angle
    next_checkpoint_id: CheckpointId
    current_lap: int
    boost_available: bool

    def next_direction(self, diff_angle: Angle) -> Angle:
        return turn_angle(self.direction, diff_angle)

    def target_point(self, diff_angle: Angle) -> Vector:
        target_distance = 5000
        next_angle = self.next_direction(diff_angle)
        return np.array([target_distance * math.cos(next_angle),
                         target_distance * math.sin(next_angle)]) + self.position


# TODO - depreciate this
def to_entities(player: List[Vehicle], opponent: List[Vehicle]) -> Entities:
    entities = Entities.empty(size=4)
    for i, v in enumerate(itertools.chain(player, opponent)):
        entities.positions[i] = v.position
        entities.speeds[i] = v.speed
        entities.directions[i] = v.direction
        entities.masses[i] = VEHICLE_MASS
        entities.next_checkpoint_id[i] = v.next_checkpoint_id
        entities.current_lap[i] = v.current_lap
        entities.boost_available[i] = v.boost_available
    return entities


"""
------------------------------------------------------------------------------------------------------------------------
ACTIONS
------------------------------------------------------------------------------------------------------------------------
"""


class OutputAction(NamedTuple):
    target: Vector
    thrust: Thrust

    def is_shield(self):
        return self.thrust < 0

    def is_boost(self):
        return self.thrust > THRUST_STRENGTH

    def __repr__(self):
        x, y = self.target
        if self.is_shield():
            thrust = "SHIELD"
        elif self.is_boost():
            thrust = "BOOST"
        else:
            thrust = str(int(self.thrust))
        return str(int(x)) + " " + str(int(y)) + " " + thrust


class Action(NamedTuple):
    angle: Angle
    thrust: Thrust

    def is_shield(self):
        return self.thrust < 0

    def is_boost(self):
        return self.thrust > THRUST_STRENGTH

    def to_output(self, vehicle: Vehicle):
        return OutputAction(target=vehicle.target_point(self.angle), thrust=self.thrust)


"""
------------------------------------------------------------------------------------------------------------------------
TRACK
------------------------------------------------------------------------------------------------------------------------
"""


class Track:
    def __init__(self, checkpoints: List[Checkpoint], total_laps: int):
        self.checkpoints = checkpoints
        self.total_checkpoints = checkpoints * (total_laps + 1)  # TODO - Hack - due to starting to CP 1!
        self.distances = np.zeros(len(self.total_checkpoints))
        self._pre_compute_distances_to_end()

    def __len__(self):
        return len(self.checkpoints)

    def progress_index(self, current_lap: int, next_checkpoint_id: int) -> int:
        return next_checkpoint_id + current_lap * len(self.checkpoints)

    def remaining_distance2(self, current_lap: int, next_checkpoint_id: int, position: Vector) -> float:
        checkpoint_id = self.progress_index(current_lap, next_checkpoint_id)
        return distance2(position, self.total_checkpoints[checkpoint_id]) + self.distances[checkpoint_id]

    def next_checkpoint(self, current_lap: int, next_checkpoint_id: int) -> Checkpoint:
        checkpoint_id = self.progress_index(current_lap, next_checkpoint_id)
        return self.total_checkpoints[checkpoint_id]

    def angle_next_checkpoint(self, vehicle: Vehicle) -> float:
        to_next_checkpoint = self.total_checkpoints[vehicle.next_checkpoint_id] - vehicle.position
        return get_angle(to_next_checkpoint)

    def _pre_compute_distances_to_end(self):
        # Compute the distance to the end: you cannot just compute to next else IA might refuse to cross a checkpoint
        for i in reversed(range(len(self.total_checkpoints) - 1)):
            distance_to_next = distance2(self.total_checkpoints[i], self.total_checkpoints[i + 1])
            self.distances[i] = self.distances[i + 1] + distance_to_next


"""
------------------------------------------------------------------------------------------------------------------------
GAME MECHANICS (Movement & Collisions)
------------------------------------------------------------------------------------------------------------------------
"""


def normal_of(p1: Vector, p2: Vector) -> Vector:
    n = np.array([p1[1] - p2[1], p2[0] - p1[0]], dtype=np.float64)
    return n / norm(n)


def find_collision(entities: Entities, i1: int, i2: int, dt: float) -> float:
    # Change referential to i1 => subtract speed of i1 to i2
    # The goal will be to check if p1 intersects p2-p3
    p1 = entities.positions[i1]
    p2 = entities.positions[i2]
    speed = entities.speeds[i2] - entities.speeds[i1]
    p3 = p2 + speed * dt

    # Quick collision check: check the distances
    # TODO - find a way to limit the computation (based on the direction of speed?)

    # Check the distance of p1 to p2-p3
    d23 = distance2(p2, p3)
    n = normal_of(p2, p3)
    dist_to_segment = abs(np.dot(n, p1 - p2))
    sum_radius = FORCE_FIELD_RADIUS * 2
    if dist_to_segment > sum_radius:
        return float('inf')

    # Find the point of intersection (a bit of trigonometry and pythagoras involved)
    distance_to_normal = np.dot(p1 - p2, p3 - p2) / math.sqrt(d23)
    distance_to_intersection: float = distance_to_normal - math.sqrt(sum_radius ** 2 - dist_to_segment ** 2)
    return distance_to_intersection / norm(speed)


def find_first_collision(entities: Entities, last_collisions: Set[Tuple[int, int]], dt: float = 1.0) -> Tuple[
    int, int, float]:
    low_t = float('inf')
    best_i = 0
    best_j = 0
    n = len(entities)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (i, j) not in last_collisions:
                t = find_collision(entities, i, j, dt)
                # TODO - some collisions are found with t < 0... should not be the case
                if t >= 0. and t < low_t:
                    low_t = t
                    best_i = i
                    best_j = j
    return best_i, best_j, low_t


def move_time_forward(entities: Entities, dt: float = 1.0):
    entities.positions += entities.speeds * dt


def bounce(entities: Entities, i1: int, i2: int, min_impulsion: float):
    # Getting the masses
    m1 = entities.masses[i1]
    m2 = entities.masses[i2]
    mcoeff = (m1 + m2) / (m1 * m2)

    # Difference of position and speeds
    dp12 = entities.positions[i2] - entities.positions[i1]
    dv12 = entities.speeds[i2] - entities.speeds[i1]

    # Computing the force
    product = np.dot(dp12, dv12)
    d12_squared = np.dot(dp12, dp12)
    f12 = dp12 * product / (d12_squared * mcoeff)

    # Apply the force (first time)
    entities.speeds[i1] += f12 / m1
    entities.speeds[i2] -= f12 / m2

    # Minimum impulsion
    norm_f = norm(f12)
    if norm_f < min_impulsion:
        f12 *= min_impulsion / norm_f

    # Apply the force (second time)
    entities.speeds[i1] += f12 / m1
    entities.speeds[i2] -= f12 / m2


def simulate_round(entities: Entities, dt: float = 1.0):
    # Run the turn to completion taking into account collisions
    last_collisions = set()
    while dt > 0.:
        i, j, t = find_first_collision(entities, last_collisions, dt)
        if t > dt:
            move_time_forward(entities, dt)
            dt = 0.
        else:
            # debug("collision:", i, j, t)
            if t > 0.:
                last_collisions.clear()
            move_time_forward(entities, t)
            bounce(entities, i, j, min_impulsion=120.)
            last_collisions.add((i, j))
            dt -= t

    # Rounding of the positions & speeds
    np.round(entities.positions, out=entities.positions)
    np.trunc(entities.speeds * 0.85, out=entities.speeds)


def apply_actions(entities: Entities, actions: List[Tuple[Thrust, Angle]]):
    # Assume my vehicles are the first 2 entities
    for i, (thrust, diff_angle) in enumerate(actions):
        if thrust > 0.:  # Movement
            entities.directions[i] = turn_angle(entities.directions[i], diff_angle)
            dv_dt = np.array([thrust * math.cos(entities.directions[i]),
                              thrust * math.sin(entities.directions[i])])
            entities.speeds[i] += dv_dt * 1.0
            entities.masses[i] = 1.
        elif thrust < 0.:  # Shield
            entities.masses[i] = 10.


def update_checkpoints(track: Track, entities: Entities):
    for i in range(len(entities)):
        new_current_lap = entities.current_lap[i]
        new_next_checkpoint_id = entities.next_checkpoint_id[i]
        next_total_checkpoint_id = new_next_checkpoint_id + new_current_lap * len(track.checkpoints)
        next_checkpoint = track.total_checkpoints[next_total_checkpoint_id]
        distance_to_checkpoint = distance2(entities.positions[i], next_checkpoint)
        if distance_to_checkpoint < CHECKPOINT_RADIUS ** 2:
            new_next_checkpoint_id += 1
            if new_next_checkpoint_id >= len(track.checkpoints):
                entities.next_checkpoint_id[i] = 0
                entities.current_lap[i] += 1


def simulate_turns(track: Track, entities: Entities, actions_by_turn: List[List[Tuple[Thrust, Angle]]]):
    for actions in actions_by_turn:
        apply_actions(entities, actions)
        simulate_round(entities, dt=1.0)
        update_checkpoints(track, entities)  # TODO - ideally, should be included in the collisions


"""
------------------------------------------------------------------------------------------------------------------------
GAME STATE
------------------------------------------------------------------------------------------------------------------------
"""


class PlayerState:
    def __init__(self):
        self.prev_checkpoints = np.array([1, 1])
        self.laps = np.array([0, 0])
        self.boost_available = np.array([True, True])
        self.shield_timeout = np.array([0, 0])

    def track_lap(self, player: List[Vehicle]):
        for i in range(len(self.shield_timeout)):
            self.shield_timeout[i] = max(0, self.shield_timeout[i] - 1)
        for i, vehicle in enumerate(player):
            if vehicle.next_checkpoint_id == 0 and self.prev_checkpoints[i] > 0:
                self.laps[i] += 1
            self.prev_checkpoints[i] = vehicle.next_checkpoint_id

    def notify_boost_used(self, vehicle_id: int):
        self.boost_available[vehicle_id] = False

    def notify_shield_used(self, vehicle_id: int):
        self.shield_timeout[vehicle_id] = 3

    def complete_vehicles(self, vehicles: List[Vehicle]):
        for vehicle_id in range(len(vehicles)):
            vehicles[vehicle_id] = self._complete_vehicle(vehicles[vehicle_id], vehicle_id)

    def _complete_vehicle(self, vehicle: Vehicle, vehicle_id: int) -> Vehicle:
        return vehicle._replace(
            current_lap=self.laps[vehicle_id],
            boost_available=self.boost_available[vehicle_id])


class GameState:
    def __init__(self):
        self.player = PlayerState()
        self.opponent = PlayerState()

    def track_lap(self, player: List[Vehicle], opponent: List[Vehicle]):
        self.player.track_lap(player)
        self.opponent.track_lap(opponent)


"""
------------------------------------------------------------------------------------------------------------------------
Agent that just tries to minimize the distance, not taking into account collisions
------------------------------------------------------------------------------------------------------------------------
"""


class GeneticAgent:
    def __init__(self, track: Track):
        self.track = track
        self.game_state = GameState()
        self.predictions: Entities = None
        self.moves = np.array([
            (BOOST_STRENGTH, 0),
            (THRUST_STRENGTH, 0),
            (THRUST_STRENGTH, -MAX_TURN_RAD),
            (THRUST_STRENGTH, +MAX_TURN_RAD),
            (0.2 * THRUST_STRENGTH, -MAX_TURN_RAD),
            (0.2 * THRUST_STRENGTH, +MAX_TURN_RAD)
        ])
        self.chronometer = Chronometer()

    def get_action(self, player: List[Vehicle], opponent: List[Vehicle]) -> List[str]:
        self.chronometer.start()
        self._complete_vehicles(player, opponent)
        actions = self._find_best_actions(player, opponent)
        debug("Time spent:", self.chronometer.spent())
        return actions

    def _complete_vehicles(self, player: List[Vehicle], opponent: List[Vehicle]):
        self.game_state.track_lap(player, opponent)
        self.game_state.player.complete_vehicles(player)
        self.game_state.opponent.complete_vehicles(opponent)

    def _find_best_actions(self, player: List[Vehicle], opponent: List[Vehicle]) -> List[str]:
        entities = to_entities(player, opponent)
        debug("PLAYER ENTITIES")
        debug(entities.positions[:2])
        debug(entities.speeds[:2])
        self._report_bad_prediction(entities)

        best_actions = self._randomized_beam_search(entities)
        self.predictions = entities.clone()
        simulate_turns(self.track, self.predictions, [best_actions])

        for i, (thrust, angle) in enumerate(best_actions):
            best_actions[i] = self._select_action(i, player[i], thrust, angle)
        return best_actions

    def _randomized_beam_search(self, entities: Entities) -> List[Tuple[Thrust, Angle]]:
        nb_strand = 10
        nb_action = 4

        best_actions = None
        best_eval = float('inf')

        scenario_count = 0

        while self.chronometer.spent() < 0.8 * RESPONSE_TIME:
            scenario_count += nb_strand

            thrusts = np.random.uniform(0., 200., size=(nb_strand, nb_action, 2))
            angles = np.random.choice([-MAX_TURN_RAD, 0, MAX_TURN_RAD], replace=True, size=(nb_strand, nb_action, 2))

            # evaluation
            evaluations = []
            for i in range(nb_strand):
                actions = []
                for j in range(nb_action):
                    actions.append(list(zip(thrusts[i, j], angles[i, j])))
                simulated = entities.clone()
                simulate_turns(self.track, simulated, actions)  # TODO - just take some vectors and not pairs
                evaluations.append((i, self._eval(simulated)))

            # mutation and selection
            evaluations.sort(key=lambda t: t[1])
            for i in range(nb_strand):
                strand_index = evaluations[i][0]
                if i == 0 and evaluations[i][1] < best_eval:
                    best_eval = evaluations[i][1]
                    best_actions = [(thrusts[strand_index][0][0], angles[strand_index][0][0]),
                                    (thrusts[strand_index][0][1], angles[strand_index][0][1])]
                if i < 3:
                    j1 = np.random.choice(nb_action)
                    j2 = np.random.choice(nb_action)
                    thrusts[strand_index][j1][0] += np.random.uniform(-20., 20.)
                    thrusts[strand_index][j2][1] += np.random.uniform(-20., 20.)
                    # TODO - angle mutations?
                else:
                    thrusts[strand_index] = np.random.uniform(0., 200., size=(nb_action, 2))
                    angles[strand_index] = np.random.choice([-MAX_TURN_RAD, 0, MAX_TURN_RAD], replace=True, size=(nb_action, 2))

        debug("count scenarios:", scenario_count)
        return best_actions

    def _eval(self, entities: Entities) -> float:
        player_dist = sum(self.track.remaining_distance2(entities.current_lap[i], entities.next_checkpoint_id[i],
                                                         entities.positions[i]) for i in range(2))
        opponent_dist = min(self.track.remaining_distance2(entities.current_lap[i], entities.next_checkpoint_id[i],
                                                           entities.positions[i]) for i in range(2, 4))
        return player_dist - opponent_dist

    def _select_action(self, vehicle_id: int, vehicle: Vehicle, best_thrust: Thrust, best_angle: Angle) -> str:
        action = Action(angle=best_angle, thrust=best_thrust)
        if action.is_boost():
            self.game_state.player.notify_boost_used(vehicle_id)
        return str(action.to_output(vehicle))

    def _report_bad_prediction(self, entities: Entities):
        if self.predictions is None:
            return

        isBad = distance2(entities.positions[:2], self.predictions.positions[:2]).sum() >= 5.
        isBad |= distance2(entities.speeds[:2], self.predictions.speeds[:2]).sum() >= 5.
        isBad |= distance2(entities.directions[:2], self.predictions.directions[:2]).sum() >= 1.
        if isBad:
            debug("BAD PREDICTION")
            debug("-" * 20)
            debug("PRED positions:", self.predictions.positions[:2])
            debug("GOT positions:", entities.positions[:2])
            debug("PRED speeds:", self.predictions.speeds[:2])
            debug("GOT speeds:", entities.speeds[:2])
            debug("PRED angles:", self.predictions.directions[:2])
            debug("GOT angles:", entities.directions[:2])


"""
------------------------------------------------------------------------------------------------------------------------
INPUT ACQUISITION
------------------------------------------------------------------------------------------------------------------------
"""


def read_checkpoint() -> Checkpoint:
    return np.array([int(j) for j in input().split()])


def read_checkpoints() -> List[Checkpoint]:
    checkpoint_count = int(input())
    return [read_checkpoint() for _ in range(checkpoint_count)]


# TODO - completely rework this: you want the entities from the start
def read_vehicle() -> Vehicle:
    x, y, vx, vy, angle, next_check_point_id = [int(j) for j in input().split()]
    return Vehicle(
        position=np.array([x, y], dtype=np.float64),
        speed=np.array([vx, vy], dtype=np.float64),
        direction=angle / 360 * 2 * math.pi,
        next_checkpoint_id=next_check_point_id,
        current_lap=0,
        boost_available=False)


"""
------------------------------------------------------------------------------------------------------------------------
GAME LOOP
------------------------------------------------------------------------------------------------------------------------
"""


# TODO - add the SHIELD state + use (but first add the collisions)


def with_first_turn_orientation(track: Track, vehicle: Vehicle) -> Vehicle:
    # For the first turn, a pod can go in any direction, here we turn it toward the goal (forbids some moves though)
    return vehicle._replace(direction=track.angle_next_checkpoint(vehicle))


def game_loop():
    total_laps = int(input())
    checkpoints = read_checkpoints()
    track = Track(checkpoints, total_laps=total_laps)
    agent = GeneticAgent(track)

    debug("laps", total_laps)
    debug("checkpoints:", checkpoints)

    for turn_nb in itertools.count(start=0, step=1):
        player_vehicles = [read_vehicle() for _ in range(2)]
        opponent_vehicles = [read_vehicle() for _ in range(2)]
        if turn_nb == 0:
            player_vehicles = [with_first_turn_orientation(track, v) for v in player_vehicles]
            opponent_vehicles = [with_first_turn_orientation(track, v) for v in opponent_vehicles]

        actions = agent.get_action(player_vehicles, opponent_vehicles)
        for action in actions:
            print(action)


if __name__ == '__main__':
    game_loop()
