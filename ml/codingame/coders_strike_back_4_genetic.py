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

VEHICLE_RADIUS = 400
VEHICLE_MASS = 1.0
VEHICLE_FRICTION = 0.85
VEHICLE_MAX_THRUST = 200
VEHICLE_BOOST_STRENGTH = 650

MAX_VEHICLE_SPEED = VEHICLE_MAX_THRUST / (1 - VEHICLE_FRICTION)    # Solve fixed point max_thrust + friction * x = x

MAX_TURN_DEG = 18
MAX_TURN_RAD = MAX_TURN_DEG / 360 * 2 * math.pi

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
        return "positions:\n" + repr(self.positions) \
             + "\nspeeds:\n" + repr(self.speeds) \
             + "\ndirections:\n" + repr(self.directions) \
             + "\nnext_progress_id:\n" + repr(self.next_progress_id) + "\n"


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
        return self.thrust > VEHICLE_MAX_THRUST

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
        self.squared_distances = np.zeros(len(self.total_checkpoints))
        self._pre_compute_distances_to_end()

    def __len__(self):
        return len(self.checkpoints)

    def get_progress_id(self, current_lap: int, next_checkpoint_id: int) -> int:
        return next_checkpoint_id + current_lap * len(self.checkpoints)

    def remaining_distance2(self, entities: Entities, vehicle_id: int) -> float:
        position = entities.positions[vehicle_id]
        progress_id = entities.next_progress_id[vehicle_id]
        return distance2(position, self.total_checkpoints[progress_id]) + self.squared_distances[progress_id]

    def next_checkpoint(self, progress_id: int) -> Checkpoint:
        return self.total_checkpoints[progress_id]

    def angle_next_checkpoint(self, progress_id: int, position: Vector) -> float:
        to_next_checkpoint = self.total_checkpoints[progress_id] - position
        return get_angle(to_next_checkpoint)

    def _pre_compute_distances_to_end(self):
        # Compute the distance to the end: you cannot just compute to next else IA might refuse to cross a checkpoint
        for i in reversed(range(len(self.total_checkpoints) - 1)):
            distance_to_next = distance2(self.total_checkpoints[i], self.total_checkpoints[i + 1])
            self.squared_distances[i] = self.squared_distances[i + 1] + distance_to_next


"""
------------------------------------------------------------------------------------------------------------------------
GAME MECHANICS (Movement & Collisions)
------------------------------------------------------------------------------------------------------------------------
"""


def normal_of(v: Vector) -> Vector:
    return np.array([-v[1], v[0]], dtype=np.float64)


def find_collision(p1: Vector, p2: Vector, speed2: Vector, sum_radius: float) -> float:
    """
    Check if there is an intersection between fixed point p1, and moving point p2
    You should change referential before calling this procedure
    """

    # Quick collision check: speed in wrong direction
    v12 = p1 - p2
    if np.dot(speed2, v12) <= 0. and norm2(v12) >= sum_radius ** 2:
        return float('inf')

    # Check the distance of p1 to segment p2-p3 (where p3 is p2 + speed)
    d23 = norm(speed2)
    n = normal_of(speed2) / d23
    dist_to_segment_2 = np.dot(n, v12) ** 2
    if dist_to_segment_2 >= sum_radius ** 2:
        return float('inf')

    # Find the point of intersection (a bit of trigonometry and pythagoras involved)
    distance_to_segment = np.dot(v12, speed2) / d23
    distance_to_intersection: float = distance_to_segment - math.sqrt(sum_radius ** 2 - dist_to_segment_2)
    return distance_to_intersection / d23


def find_cp_collision(track: Track, entities: Entities, i: int, dt: float) -> float:
    p1 = track.next_checkpoint(entities.next_progress_id[i])
    p2 = entities.positions[i]
    speed2 = entities.speeds[i] * dt
    if distance2(p1, p2) > MAX_VEHICLE_SPEED ** 2:  # Useful optimization
        return float('inf')
    return find_collision(p1, p2, speed2, sum_radius=CHECKPOINT_RADIUS)


def find_unit_collision(entities: Entities, i1: int, i2: int, dt: float) -> float:
    # Change referential to i1 => subtract speed of i1 to i2 (the goal will be to check if p1 intersects p2-p3)
    p1 = entities.positions[i1]
    p2 = entities.positions[i2]
    speed2 = (entities.speeds[i2] - entities.speeds[i1]) * dt
    return find_collision(p1, p2, speed2, sum_radius=VEHICLE_RADIUS * 2)


def find_first_collision(track: Track, entities: Entities,
                         last_collisions: Set[Tuple[int, int]],
                         dt: float = 1.0) -> Tuple[int, int, float]:
    low_t = dt * 1.1
    best_i = best_j = 0
    n = len(entities)
    for i in range(n):
        t = find_cp_collision(track, entities, i, dt)
        if 0. <= t < low_t:
            low_t = t
            best_i = i
            best_j = -1     # Indicates checkpoint
        for j in range(i + 1, n):
            if (i, j) not in last_collisions:
                t = find_unit_collision(entities, i, j, dt)
                if 0. <= t < low_t:
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


def simulate_movements(track: Track, entities: Entities, dt: float = 1.0):
    # Run the turn to completion taking into account collisions
    last_collisions = set()
    while dt > 0.:
        i, j, t = find_first_collision(track, entities, last_collisions, dt)
        if t > dt:
            move_time_forward(entities, dt)
            dt = 0.
        else:
            if t > 0.:
                last_collisions.clear()
            move_time_forward(entities, t)
            if j >= 0:
                bounce(entities, i, j, min_impulsion=120.)  # Collision with unit
            else:
                entities.next_progress_id[i] += 1           # Collision with checkpoint
            last_collisions.add((i, j))
            dt -= t

    # Rounding of the positions & speeds
    np.round(entities.positions, out=entities.positions)
    np.trunc(entities.speeds * VEHICLE_FRICTION, out=entities.speeds)


def apply_actions(entities: Entities, thrusts: np.ndarray, diff_angles: np.ndarray):
    # SHAPE of thrusts/diff_angles should be: (nb_entities, )
    # Assume my vehicles are the first 2 entities
    for i in range(thrusts.shape[0]):
        thrust = thrusts[i]
        diff_angle = diff_angles[i]
        entities.shield_timeout[i] = max(0, entities.shield_timeout[i] - 1)
        entities.directions[i] = mod_angle(entities.directions[i] + diff_angle)
        if thrust > 0.:  # Movement
            if entities.shield_timeout[i] == 0:  # No thrusts in case of shield
                entities.speeds[i][0] += thrust * math.cos(entities.directions[i])
                entities.speeds[i][1] += thrust * math.sin(entities.directions[i])
        elif thrust < 0.:  # Shield
            entities.shield_timeout[i] = 3


def simulate_turns(track: Track, entities: Entities, thrusts: np.ndarray, diff_angles: np.ndarray):
    # SHAPE of thrusts/diff_angles should be: (nb_turns, nb_entities)
    nb_turns, _ = thrusts.shape
    for turn_id in range(nb_turns):
        apply_actions(entities, thrusts[turn_id], diff_angles[turn_id])
        simulate_movements(track, entities, dt=1.0)


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
            entities.next_progress_id[i] = track.get_progress_id(self.laps[i], self.prev_checkpoint_id[i])
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
            (VEHICLE_BOOST_STRENGTH, 0),
            (VEHICLE_MAX_THRUST, 0),
            (VEHICLE_MAX_THRUST, -MAX_TURN_RAD),
            (VEHICLE_MAX_THRUST, +MAX_TURN_RAD),
            (0.2 * VEHICLE_MAX_THRUST, -MAX_TURN_RAD),
            (0.2 * VEHICLE_MAX_THRUST, +MAX_TURN_RAD)
        ])
        self.chronometer = Chronometer()
        self.previous_thrust_dna = None
        self.previous_angle_dna = None
        self.runner_id: int = 0
        self.opponent_runner_id: int = 2

    # TODO - alternative: do a search in depth as was done before:
    #   Objective of first drone is to win the race
    #   Objective of second drone is to block the opponent
    #   The opponent should have its move generated first

    # TODO - alternative: do a beam search starting as a tree
    #   Keep only a limited number of leaves open?
    #   Or do some kind of MCTS: explore the most favorables

    def get_action(self, entities: Entities) -> List[Action]:
        self.chronometer.start()
        self._identify_roles(entities)
        actions = self._find_best_actions(entities)
        debug("Time spent:", self.chronometer.spent())
        return actions

    def _identify_roles(self, entities: Entities):
        # TODO - try to classify the IA of the opponent in here
        remaining_distances = np.array([0.] * len(entities))
        for i in range(len(remaining_distances)):
            remaining_distances[i] = self.track.remaining_distance2(entities, i)
        self.runner_id = np.argmin(remaining_distances[:2])
        if len(entities) > 2: # TODO - hack for my own simulations
            self.opponent_runner_id = 2 + np.argmin(remaining_distances[2:])

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
        nb_entities = 2

        best_thrusts = None
        best_angles = None
        min_eval = float('inf')
        scenario_count = 0

        # TODO - The GA algorithm is too timid (slows down when it should not)

        init_thrusts, init_angles = self._initial_solution(entities, nb_action)

        thrusts = np.random.uniform(0., 200., size=(nb_strand, nb_action, nb_entities))  # TODO - encourage fast speeds
        thrusts[0] = init_thrusts
        if self.previous_thrust_dna is not None:
            thrusts[1][:-1] = self.previous_thrust_dna[1:]

        angles = np.random.uniform(-MAX_TURN_RAD, MAX_TURN_RAD, size=(nb_strand, nb_action, nb_entities))
        angles[0] = init_angles
        if self.previous_angle_dna is not None:
            angles[1][:-1] = self.previous_angle_dna[1:]

        evaluations = np.zeros(shape=nb_strand, dtype=np.float64)

        # TODO - decreasing amplitude each time we beat a record

        while self.chronometer.spent() < 0.8 * RESPONSE_TIME:
            scenario_count += nb_strand

            # Evaluation of the different solutions
            for i in range(nb_strand):
                simulated = entities.clone()
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
            thrusts[indices[5]] = np.random.uniform(0., 200., size=(nb_action, nb_entities))
            angles[indices[3]] = angles[indices[0]]
            angles[indices[4]] = best_angles
            angles[indices[5]] = np.random.uniform(-MAX_TURN_RAD, MAX_TURN_RAD, size=(nb_action, nb_entities))

            # Random mutations for every-one
            thrusts += np.random.normal(-10., 10., size=(nb_strand, nb_action, nb_entities))
            angles += np.random.normal(-MAX_TURN_RAD * 0.1, MAX_TURN_RAD * 0.1, size=(nb_strand, nb_action, nb_entities))

            # Make sure the solution are correct
            thrusts.clip(0., 200., out=thrusts)
            angles.clip(-MAX_TURN_RAD, MAX_TURN_RAD, out=angles)

        debug("count scenarios:", scenario_count)
        return best_thrusts, best_angles

    def _initial_solution(self, entities: Entities, nb_action: int) -> Tuple[np.ndarray, np.ndarray]:
        # Solution based on a kind of PID: improve it
        nb_entities = 2
        entities = entities.clone()
        thrusts = np.zeros(shape=(nb_action, nb_entities))
        diff_angles = np.zeros(shape=(nb_action, nb_entities))
        for d in range(nb_action):
            for i in range(nb_entities):  # TODO - do this for the opponent as well
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
        my_perturbator = 1 - self.runner_id
        my_dist = self.track.remaining_distance2(entities, self.runner_id)
        if len(entities) > self.opponent_runner_id: # TODO - hack for my own simulation
            his_dist = self.track.remaining_distance2(entities, self.opponent_runner_id)
            closing_dist = distance2(entities.positions[my_perturbator], entities.positions[self.opponent_runner_id])
        else:
            his_dist = 0
            closing_dist = 0

        '''
        closing_dist = distance2(entities.positions[my_perturbator],
                                 self.track.next_checkpoint(entities.next_progress_id[self.opponent_runner_id] + 1))
        '''

        # TODO - add a term to encourage aggressive attacks (shocks at high speed)
        # TODO - encourage to move the next checkpoint of HIS runner
        # TODO - encourage to make sure the opponent does not get close to HIS next cp?
        go_fast = norm2(entities.speeds[self.runner_id])
        return my_dist - his_dist + 0.1 * closing_dist + go_fast

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
