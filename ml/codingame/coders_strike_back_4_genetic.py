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
    return math.sqrt(np.dot(v, v))


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
    if diff_angle > MAX_TURN_RAD:
        diff_angle = MAX_TURN_RAD
    elif diff_angle < -MAX_TURN_RAD:
        diff_angle = -MAX_TURN_RAD
    return mod_angle(prev_angle + diff_angle)


def target_point(position: Vector, prev_angle: Angle, diff_angle: Angle) -> Vector:
    next_angle = turn_angle(prev_angle, diff_angle)
    return np.array([position[0] + 2000 * math.cos(next_angle),
                     position[1] + 2000 * math.sin(next_angle)])


@dataclass(frozen=False)
class Entities:
    length: int
    positions: np.ndarray
    speeds: np.ndarray
    shield_timeout: np.ndarray      # Include the notion of mass for PODs
    directions: np.ndarray
    next_progress_id: np.ndarray
    boost_available: np.ndarray

    @classmethod
    def empty(cls, size: int):
        return Entities(
            length=size,
            positions=np.zeros(shape=(size, 2)),
            speeds=np.zeros(shape=(size, 2)),
            shield_timeout=np.zeros(shape=size, dtype=np.int64),
            directions=np.zeros(shape=size),
            next_progress_id=np.zeros(shape=size, dtype=np.int64),
            boost_available=np.zeros(shape=size, dtype=bool))

    def __len__(self):
        return self.length

    def __eq__(self, other):
        return np.array_equal(self.positions, other.positions) \
               and np.array_equal(self.speeds, other.speeds) \
               and np.array_equal(self.directions, other.directions) \
               and np.array_equal(self.next_progress_id, other.next_progress_id)

    def clone(self):
        return Entities(
            length=self.length,
            positions=self.positions.copy(),
            speeds=self.speeds.copy(),
            directions=self.directions.copy(),
            next_progress_id=self.next_progress_id.copy(),
            boost_available=self.boost_available.copy(),
            shield_timeout=self.shield_timeout.copy())

    def __repr__(self):
        return "positions:\n" + repr(self.positions) + "\nspeeds:\n" + repr(self.speeds) + "\n"


"""
------------------------------------------------------------------------------------------------------------------------
ACTIONS
------------------------------------------------------------------------------------------------------------------------
"""


class Action(NamedTuple):
    angle: Angle
    thrust: Thrust

    def is_shield(self):
        return self.thrust < 0

    def is_boost(self):
        return self.thrust > THRUST_STRENGTH

    def to_str(self, position: Vector, prev_angle: Angle) -> str:
        x, y = target_point(position, prev_angle, self.angle)
        if self.is_shield():
            thrust = "SHIELD"
        elif self.is_boost():
            thrust = "BOOST"
        else:
            thrust = str(int(self.thrust))
        return str(int(x)) + " " + str(int(y)) + " " + thrust


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

    def get_progress_index(self, current_lap: int, next_checkpoint_id: int) -> int:
        return next_checkpoint_id + current_lap * len(self.checkpoints)

    def remaining_distance2(self, progress_index: int, position: Vector) -> float:
        return distance2(position, self.total_checkpoints[progress_index]) + self.distances[progress_index]

    def next_checkpoint(self, progress_index: int) -> Checkpoint:
        return self.total_checkpoints[progress_index]

    def angle_next_checkpoint(self, progress_index: int, position: Vector) -> float:
        to_next_checkpoint = self.total_checkpoints[progress_index] - position
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


def normal_of(v: Vector) -> Vector:
    return np.array([-v[1], v[0]], dtype=np.float64)


def find_collision(entities: Entities, i1: int, i2: int, dt: float) -> float:
    # Change referential to i1 => subtract speed of i1 to i2
    # The goal will be to check if p1 intersects p2-p3
    p1 = entities.positions[i1]
    p2 = entities.positions[i2]
    speed = (entities.speeds[i2] - entities.speeds[i1]) * dt

    # Quick collision check: no speed
    d23 = norm(speed)
    if d23 == 0. and distance2(p1, p2) > FORCE_FIELD_RADIUS ** 2:
        return float('inf')

    # TODO - find other ways to limit the computation (based on the direction of speed?)
    # TODO - if speed does not go in right direction: then you could find a t < 0. => forbid this

    # Check the distance of p1 to segment p2-p3 (where p3 is p2 + speed)
    n = normal_of(speed) / d23
    dist_to_segment = abs(np.dot(n, p1 - p2))
    sum_radius = FORCE_FIELD_RADIUS * 2
    if dist_to_segment > sum_radius:
        return float('inf')

    # Find the point of intersection (a bit of trigonometry and pythagoras involved)
    distance_to_segment = np.dot(p1 - p2, speed) / d23
    distance_to_intersection: float = distance_to_segment - math.sqrt(sum_radius ** 2 - dist_to_segment ** 2)
    return distance_to_intersection / d23


def find_first_collision(entities: Entities, last_collisions: Set[Tuple[int, int]],
                         dt: float = 1.0) -> Tuple[int, int, float]:
    low_t = float('inf')
    best_i = best_j = 0
    n = len(entities)
    for i in range(n - 1):
        # TODO - just check the collision with YOUR checkpoint - would limit the amount of impacts on performance
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
    m1 = 1. if entities.shield_timeout[i1] < 3 else 10.
    m2 = 1. if entities.shield_timeout[i2] < 3 else 10.
    mcoeff = (m1 + m2) / (m1 * m2)

    # Difference of position and speeds
    dp12 = entities.positions[i2] - entities.positions[i1]
    dv12 = entities.speeds[i2] - entities.speeds[i1]

    # Computing the force
    d12_squared = np.dot(dp12, dp12)
    f12 = dp12 * np.dot(dp12, dv12) / (d12_squared * mcoeff)

    # Apply half the force (first time)
    entities.speeds[i1] += f12 / m1
    entities.speeds[i2] -= f12 / m2

    # Minimum half-impulsion
    norm_f = norm(f12)
    if norm_f < min_impulsion:
        f12 *= min_impulsion / norm_f

    # Apply half the force (second time)
    entities.speeds[i1] += f12 / m1
    entities.speeds[i2] -= f12 / m2


def simulate_movements(entities: Entities, dt: float = 1.0):
    # Run the turn to completion taking into account collisions
    last_collisions = set()
    while dt > 0.:
        i, j, t = find_first_collision(entities, last_collisions, dt)
        if t > dt:
            move_time_forward(entities, dt)
            dt = 0.
        else:
            if t > 0.:
                last_collisions.clear()
            move_time_forward(entities, t)
            bounce(entities, i, j, min_impulsion=120.)
            last_collisions.add((i, j))
            dt -= t

    # Rounding of the positions & speeds
    np.round(entities.positions, out=entities.positions)
    np.trunc(entities.speeds * 0.85, out=entities.speeds)


def apply_actions(entities: Entities, thrusts: np.ndarray, diff_angles: np.ndarray):
    # SHAPE of thrusts/diff_angles should be: (nb_entities, )
    # Assume my vehicles are the first 2 entities
    for i in range(thrusts.shape[0]):
        thrust = thrusts[i]
        diff_angle = diff_angles[i]
        entities.shield_timeout[i] -= 1
        if thrust > 0.:  # Movement
            # TODO - disable the thrust if the timeout is positive
            entities.directions[i] = turn_angle(entities.directions[i], diff_angle) # TODO - clip already done? Then drop the "if" in turn_angle
            entities.speeds[i][0] += thrust * math.cos(entities.directions[i])
            entities.speeds[i][1] += thrust * math.sin(entities.directions[i])
        elif thrust < 0.:  # Shield
            entities.shield_timeout[i] = 3


def update_checkpoints(track: Track, entities: Entities):
    for i in range(len(entities)):
        cp_progress_id = entities.next_progress_id[i]
        next_checkpoint = track.next_checkpoint(cp_progress_id)
        distance_to_checkpoint = distance2(entities.positions[i], next_checkpoint)
        if distance_to_checkpoint < CHECKPOINT_RADIUS ** 2:
            entities.next_progress_id[i] += 1


def simulate_turns(track: Track, entities: Entities, thrusts: np.ndarray, diff_angles: np.ndarray):
    # SHAPE of thrusts/diff_angles should be: (nb_turns, nb_entities)
    nb_turns, _ = thrusts.shape
    for turn_id in range(nb_turns):
        apply_actions(entities, thrusts[turn_id], diff_angles[turn_id])
        simulate_movements(entities, dt=1.0)
        update_checkpoints(track, entities)  # TODO - ideally, should be included each time there is a collision: agent too cautious now


"""
------------------------------------------------------------------------------------------------------------------------
GAME STATE
------------------------------------------------------------------------------------------------------------------------
"""


class GameState:
    def __init__(self):
        self.prev_checkpoint_id = np.array([1] * 4)
        self.laps = np.array([0] * 4)
        self.boost_available = np.array([True] * 4)
        self.shield_timeout = np.array([0] * 4)

    def track_lap(self, vehicle_id: int, next_checkpoint_id: int):
        self.shield_timeout[vehicle_id] = max(0, self.shield_timeout[vehicle_id] - 1)
        if next_checkpoint_id == 0 and self.prev_checkpoint_id[vehicle_id] > 0:
            self.laps[vehicle_id] += 1
        self.prev_checkpoint_id[vehicle_id] = next_checkpoint_id

    def complete_vehicles(self, entities: Entities, track: Track):
        for i in range(len(entities)):
            entities.next_progress_id[i] = track.get_progress_index(self.laps[i], self.prev_checkpoint_id[i])
            entities.boost_available[i] = self.boost_available[i]
            entities.shield_timeout[i] = self.shield_timeout[i]

    def notify_boost_used(self, vehicle_id: int):
        self.boost_available[vehicle_id] = False

    def notify_shield_used(self, vehicle_id: int):
        self.shield_timeout[vehicle_id] = 3


"""
------------------------------------------------------------------------------------------------------------------------
AGENT
------------------------------------------------------------------------------------------------------------------------
"""


class GeneticAgent:
    def __init__(self, track: Track):
        self.track = track
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
        self.previous_thrust_dna = None
        self.previous_angle_dna = None

    # TODO - alternative: do a search in depth as was done before:
    #   Objective of first drone is to win the race
    #   Objective of second drone is to block the opponent
    #   The opponent should have its move generated first

    # TODO - alternative: do a beam search starting as a tree
    #   Keep only a limited number of leaves open?
    #   Or do some kind of MCTS: explore the most favorables

    def get_action(self, entities: Entities) -> List[Action]:
        self.chronometer.start()
        actions = self._find_best_actions(entities)
        debug("Time spent:", self.chronometer.spent())
        return actions

    def _find_best_actions(self, entities: Entities) -> List[Action]:
        self._report_bad_prediction(entities)

        thrust_dna, angle_dna = self._randomized_beam_search(entities)
        self.previous_thrust_dna = thrust_dna
        self.previous_angle_dna = angle_dna

        self.predictions = entities.clone()
        simulate_turns(self.track, self.predictions, thrust_dna[:1], angle_dna[:1])
        return [Action(angle=angle_dna[0][i], thrust=thrust_dna[0][i]) for i in range(2)]

    def _randomized_beam_search(self, entities: Entities) -> Tuple[np.ndarray, np.ndarray]:
        nb_strand = 6
        nb_action = 6

        best_thrusts = None
        best_angles = None
        min_eval = float('inf')
        scenario_count = 0

        init_thrusts, init_angles = self._initial_solution(entities, nb_action)

        thrusts = np.random.uniform(0., 200., size=(nb_strand, nb_action, 2))
        thrusts[0] = init_thrusts
        if self.previous_thrust_dna is not None:
            thrusts[1][:-1] = self.previous_thrust_dna[1:]

        angles = np.random.choice([-MAX_TURN_RAD, 0, MAX_TURN_RAD], replace=True, size=(nb_strand, nb_action, 2))
        angles[0] = init_angles
        if self.previous_angle_dna is not None:
            angles[1][:-1] = self.previous_angle_dna[1:]

        evaluations = np.zeros(shape=nb_strand, dtype=np.float64)

        while self.chronometer.spent() < 0.8 * RESPONSE_TIME:
            scenario_count += nb_strand

            # Make sure the solution are correct
            thrusts.clip(0., 200., out=thrusts)
            angles.clip(-MAX_TURN_RAD, MAX_TURN_RAD, out=angles)

            # Evaluation of the different solutions
            for i in range(nb_strand):
                simulated = entities.clone()
                simulated.length = 2  # TODO - Ignore the opponents as long as we do not predict them
                simulate_turns(self.track, simulated, thrusts[i], angles[i])
                evaluations[i] = self._eval(simulated)

            # Keeping track of the best overall solution
            indices = np.argsort(evaluations)
            best_index = indices[0]
            if evaluations[best_index] < min_eval:
                min_eval = evaluations[best_index]
                best_thrusts = thrusts[best_index].copy()
                best_angles = angles[best_index].copy()

            # Selections of the best + re-injection of the current best so far
            thrusts[indices[3]] = thrusts[indices[0]]
            thrusts[indices[4]] = best_thrusts
            thrusts[indices[5]] = np.random.uniform(0., 200., size=(nb_action, 2))
            angles[indices[3]] = angles[indices[0]]
            angles[indices[4]] = best_angles
            angles[indices[5]] = np.random.choice([-MAX_TURN_RAD, 0, MAX_TURN_RAD], replace=True, size=(nb_action, 2))

            # Random mutations for every-one
            thrusts += np.random.uniform(-20., 20., size=(nb_strand, nb_action, 2))
            angles += np.random.uniform(-MAX_TURN_RAD * 0.2, MAX_TURN_RAD * 0.2, size=(nb_strand, nb_action, 2))

        debug("count scenarios:", scenario_count)
        return best_thrusts, best_angles

    def _initial_solution(self, entities: Entities, nb_action: int) -> Tuple[np.ndarray, np.ndarray]:
        # Solution based on a kind of PID: improve it
        entities = entities.clone()
        thrusts = np.zeros(shape=(nb_action, 2))
        diff_angles = np.zeros(shape=(nb_action, 2))
        for d in range(nb_action):
            for i in range(2):  # TODO - do this for the opponent as well
                p = entities.positions[i]
                s = entities.speeds[i]
                c = self.track.next_checkpoint(entities.next_progress_id[i])
                cp_angle = mod_angle(get_angle(c - p - 2 * s))
                dir_angle = entities.directions[i]
                diff_angle = mod_angle(cp_angle - dir_angle)
                if diff_angle > math.pi:
                    diff_angle = diff_angle - 2 * math.pi
                diff_angles[d][i] = diff_angle
                thrusts[d][i] = distance(p, c) - 3. * norm(s)  # TODO - NOT good when not in the right direction
                # debug("cp angle", cp_angle / math.pi * 180)
                # debug("veh angle", dir_angle / math.pi * 180)
                # debug("diff angle", diff_angle / math.pi * 180)
            thrusts.clip(0., 200., out=thrusts)
            diff_angles.clip(-MAX_TURN_RAD, MAX_TURN_RAD, out=diff_angles)
            simulate_turns(self.track, entities, thrusts[d:d + 1], diff_angles[d:d + 1])
        return thrusts, diff_angles

    def _eval(self, entities: Entities) -> float:
        remaining_distances = np.array([0.] * 4)
        for i in range(len(remaining_distances)):
            progress_index = entities.next_progress_id[i]
            remaining_distances[i] = self.track.remaining_distance2(progress_index, entities.positions[i])

        my_runner = np.argmin(remaining_distances[:2])
        my_perturbator = 1 - my_runner
        his_runner = 2 + np.argmin(remaining_distances[2:])

        my_dist = remaining_distances[my_runner]
        his_dist = remaining_distances[his_runner]
        closing_dist = distance2(entities.positions[my_perturbator], entities.positions[his_runner])
        # TODO - add a term to encourage aggressive attacks (shocks at high speed)
        return my_dist - his_dist + 0.15 * closing_dist

    def _report_bad_prediction(self, entities: Entities):
        # debug("PLAYER ENTITIES")
        # debug(entities.positions[:2])
        # debug(entities.speeds[:2])
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


def read_entities(game_state: GameState, track: Track, turn_nb: int) -> Entities:
    entities = Entities.empty(size=4)
    for vehicle_id in range(4):
        x, y, vx, vy, angle, next_check_point_id = [int(s) for s in input().split()]
        game_state.track_lap(vehicle_id, next_check_point_id)
        entities.positions[vehicle_id][0] = x
        entities.positions[vehicle_id][1] = y
        entities.speeds[vehicle_id][0] = vx
        entities.speeds[vehicle_id][1] = vy
        if turn_nb == 0:
            entities.directions[vehicle_id] = track.angle_next_checkpoint(next_check_point_id, entities.positions[vehicle_id])
        else:
            entities.directions[vehicle_id] = angle / 360 * 2 * math.pi
    game_state.complete_vehicles(entities, track)
    return entities


"""
------------------------------------------------------------------------------------------------------------------------
GAME LOOP
------------------------------------------------------------------------------------------------------------------------
"""


def serialize_action(game_state: GameState, entities: Entities, vehicle_id: int, action: Action) -> str:
    if action.is_boost():
        game_state.notify_boost_used(vehicle_id)
    elif action.is_shield():
        game_state.notify_shield_used(vehicle_id)
    return action.to_str(entities.positions[vehicle_id], entities.directions[vehicle_id])


def game_loop():
    total_laps = int(input())
    checkpoints = read_checkpoints()

    debug("laps", total_laps)
    debug("checkpoints:", checkpoints)

    track = Track(checkpoints, total_laps=total_laps)
    game_state = GameState()
    agent = GeneticAgent(track)

    for turn_nb in itertools.count(start=0, step=1):
        entities = read_entities(game_state, track, turn_nb)
        actions = agent.get_action(entities)
        for vehicle_id, action in enumerate(actions):
            print(serialize_action(game_state, entities, vehicle_id, action))


if __name__ == '__main__':
    game_loop()
