import math
import sys
import itertools
from typing import List, NamedTuple, Tuple
import time

import numpy as np


# import concurrent.futures


"""
Constants
"""

WIDTH = 16000
HEIGHT = 9000

CHECKPOINT_RADIUS = 600
FORCE_FIELD_RADIUS = 400

MAX_TURN_DEG = 18
MAX_TURN_RAD = MAX_TURN_DEG / 360 * 2 * math.pi

BOOST_STRENGTH = 650

FIRST_RESPONSE_TIME = 1000
RESPONSE_TIME = 75

TOP_LEFT = (0, 0)
BOT_RIGHT = (WIDTH - 1, HEIGHT - 1)


"""
Utils
"""


def debug(*args):
    print(*args, file=sys.stderr)


"""
Vector arithmetic
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
Data structures
"""


Checkpoint = np.ndarray
CheckpointId = int
Thrust = int


class Vehicle(NamedTuple):
    position: Vector
    speed: Vector
    direction: Angle
    next_checkpoint_id: CheckpointId
    current_lap: int
    boost_available: bool

    def next_direction(self, diff_angle: Angle) -> Angle:
        if diff_angle > 0:
            diff_angle = min(MAX_TURN_RAD, diff_angle)
        elif diff_angle < 0:
            diff_angle = max(-MAX_TURN_RAD, diff_angle)
        return mod_angle(self.direction + diff_angle)

    def target_point(self, diff_angle: Angle) -> Vector:
        target_distance = 5000
        next_angle = self.next_direction(diff_angle)
        return np.array([target_distance * math.cos(next_angle),
                         target_distance * math.sin(next_angle)]) + self.position


"""
Track
"""


class Track:
    def __init__(self, checkpoints: List[Checkpoint], total_laps: int):
        self.checkpoints = checkpoints
        self.total_checkpoints = checkpoints * (total_laps + 1)  # TODO - Hack - due to starting to CP 1!
        self.distances = np.zeros(len(self.total_checkpoints))
        self._pre_compute_distances_to_end()

    # TODO - replace 'thrust' by 'action" (boost and shield), or consider that -1 is shield

    def __len__(self):
        return len(self.checkpoints)

    def progress(self, vehicle: Vehicle) -> int:
        return vehicle.next_checkpoint_id + vehicle.current_lap * len(self.checkpoints)

    def remaining_distance2(self, vehicle: Vehicle) -> float:
        checkpoint_id = self.progress(vehicle)
        return distance2(vehicle.position, self.total_checkpoints[checkpoint_id]) + self.distances[checkpoint_id]

    def next_checkpoint(self, vehicle: Vehicle) -> Checkpoint:
        checkpoint_id = self.progress(vehicle)
        return self.total_checkpoints[checkpoint_id]

    def angle_next_checkpoint(self, vehicle: Vehicle) -> float:
        to_next_checkpoint = self.total_checkpoints[vehicle.next_checkpoint_id] - vehicle.position
        return get_angle(to_next_checkpoint)

    def next_position(self, vehicle: Vehicle, thrust: Thrust, diff_angle: Angle, dt: float = 1.0) -> Vehicle:
        new_direction = vehicle.next_direction(diff_angle)
        dv_dt = np.array([thrust * math.cos(new_direction), thrust * math.sin(new_direction)])
        new_speed = vehicle.speed + dv_dt * dt
        new_position = np.round(vehicle.position + new_speed * dt)
        new_next_checkpoint_id, new_current_lap = self._new_next_checkpoint(vehicle, new_position)
        return Vehicle(
            position=new_position,
            speed=np.trunc(new_speed * 0.85),
            direction=new_direction,
            next_checkpoint_id=new_next_checkpoint_id,
            current_lap=new_current_lap,
            boost_available=vehicle.boost_available and thrust <= 100)

    def _new_next_checkpoint(self, vehicle: Vehicle, new_position: Vector):
        new_current_lap = vehicle.current_lap
        new_next_checkpoint_id = vehicle.next_checkpoint_id
        distance_to_checkpoint = distance2(new_position, self.total_checkpoints[vehicle.next_checkpoint_id])
        if distance_to_checkpoint < CHECKPOINT_RADIUS ** 2:
            new_next_checkpoint_id += 1
            if new_next_checkpoint_id >= len(self.checkpoints):
                new_next_checkpoint_id = 0
                new_current_lap += 1
        return new_next_checkpoint_id, new_current_lap

    def _pre_compute_distances_to_end(self):
        # Compute the distance to the end: you cannot just compute to next else IA might refuse to cross a checkpoint
        for i in reversed(range(len(self.total_checkpoints) - 1)):
            distance_to_next = distance2(self.total_checkpoints[i], self.total_checkpoints[i + 1])
            self.distances[i] = self.distances[i + 1] + distance_to_next


"""
Game state
"""


class PlayerState:
    # TODO - you should track also the timer to next checkpoint... for you and the opponent

    def __init__(self, nb_checkpoints: int):
        self.prev_checkpoints = np.array([1, 1])
        self.laps = np.array([0, 0])
        self.boost_available = np.array([True, True])

    def track_lap(self, player: List[Vehicle]):
        for i, vehicle in enumerate(player):
            if vehicle.next_checkpoint_id == 0 and self.prev_checkpoints[i] > 0:
                self.laps[i] += 1
            self.prev_checkpoints[i] = vehicle.next_checkpoint_id

    def notify_boost_used(self, vehicle_id: int):
        self.boost_available[vehicle_id] = False

    def complete_vehicles(self, vehicles: List[Vehicle]):
        for vehicle_id in range(len(vehicles)):
            vehicles[vehicle_id] = self._complete_vehicle(vehicles[vehicle_id], vehicle_id)

    def _complete_vehicle(self, vehicle: Vehicle, vehicle_id: int) -> Vehicle:
        return vehicle._replace(
            current_lap=self.laps[vehicle_id],
            boost_available=self.boost_available[vehicle_id]
        )


class GameState:
    def __init__(self, nb_checkpoints: int):
        self.player = PlayerState(nb_checkpoints)
        self.opponent = PlayerState(nb_checkpoints)

    def track_lap(self, player: List[Vehicle], opponent: List[Vehicle]):
        self.player.track_lap(player)
        self.opponent.track_lap(opponent)


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
Agent that just tries to minimize the distance, not taking into account collisions
"""


class ShortestPathAgent:
    def __init__(self, track: Track):
        self.track = track
        self.game_state = GameState(nb_checkpoints=len(self.track))
        self.predictions: List[Vehicle] = [None, None]
        self.moves = np.array([
            (BOOST_STRENGTH, 0),
            (100, 0),
            (100, -MAX_TURN_RAD),
            (100, +MAX_TURN_RAD),
            (20, -MAX_TURN_RAD),
            (20, +MAX_TURN_RAD)
        ])
        self.chronometer = Chronometer()
        # TODO - there is a problem: if you put thrust=0, then the IA stay stuck in front of checkpoint sometimes

    def get_action(self, player: List[Vehicle], opponent: List[Vehicle]) -> List[str]:
        self.chronometer.start()
        self._complete_vehicles(player, opponent)
        return self._find_best_actions(player, opponent)

    def _complete_vehicles(self, player: List[Vehicle], opponent: List[Vehicle]):
        self.game_state.track_lap(player, opponent)
        self.game_state.player.complete_vehicles(player)
        self.game_state.opponent.complete_vehicles(opponent)

    def _find_best_actions(self, player: List[Vehicle], opponent: List[Vehicle]) -> List[str]:
        actions = []
        for vehicle_id, vehicle in enumerate(player):
            self._report_bad_prediction(vehicle, vehicle_id)
            if self._is_runner(player, vehicle_id):
                debug("runner:", vehicle)
                action, next_vehicle = self._shortest_path_action(vehicle, vehicle_id, metric=self.track.remaining_distance2)
            else:
                debug("follower:", vehicle)
                action, next_vehicle = self._intercept(vehicle, vehicle_id, opponent)
            self.predictions[vehicle_id] = next_vehicle
            actions.append(action)
        return actions

    def _is_runner(self, vehicles: List[Vehicle], vehicle_id: int):
        other_id = 1 - vehicle_id
        return self.track.progress(vehicles[vehicle_id]) >= self.track.progress(vehicles[other_id])

    def _intercept(self, vehicle: Vehicle, vehicle_id: int, opponents: List[Vehicle]) -> Tuple[str, Vehicle]:
        def intercept_metric(v: Vehicle):
            dist_to_opponent = distance2(v.position, opponent.position + 4 * opponent.speed)
            dist_to_opponent_dst = distance2(v.position, self.track.next_checkpoint(opponent))
            return (dist_to_opponent + dist_to_opponent_dst) / 2

        for opponent_id, opponent in enumerate(opponents):
            if self._is_runner(opponents, opponent_id):
                return self._shortest_path_action(vehicle, vehicle_id, metric=intercept_metric)

    def _shortest_path_action(self, vehicle: Vehicle, vehicle_id: int, metric) -> Tuple[str, Vehicle]:
        best_thrust = Thrust()
        best_angle = Angle()
        min_score = float('inf')
        best_next_vehicle = None
        for thrust, angle in self._possible_moves(vehicle):
            if self.chronometer.spent() > 90:
                debug("TIMEOUT: Skipping action", thrust, "with angle", angle)
                break

            next_vehicle = self.track.next_position(vehicle, thrust, angle)
            depth = 3 if next_vehicle.boost_available else 4
            score = self._explore_move(next_vehicle, metric, depth=depth)
            if score < min_score:
                min_score = score
                best_thrust = thrust
                best_angle = angle
                best_next_vehicle = next_vehicle

        debug("------------------------")
        debug("current:", vehicle)
        debug("best action:", best_thrust, "with angle", best_angle)
        debug("prediction:", best_next_vehicle)
        debug("------------------------")

        return self._select_action(vehicle_id, vehicle, best_thrust, best_angle), best_next_vehicle

    def _select_action(self, vehicle_id: int, vehicle: Vehicle, best_thrust: Thrust, best_angle: Angle):
        if best_thrust > 100:
            best_thrust = "BOOST"
            self.game_state.player.notify_boost_used(vehicle_id)
        else:
            best_thrust = str(int(best_thrust))
        next_x, next_y = vehicle.target_point(best_angle)
        return str(int(next_x)) + " " + str(int(next_y)) + " " + best_thrust

    def _explore_move(self, vehicle: Vehicle, metric, depth: int) -> float:
        if depth <= 1:
            return metric(vehicle)

        min_score = float('inf')
        for thrust, angle in self._possible_moves(vehicle):
            next_vehicle = self.track.next_position(vehicle, thrust, angle, dt=1.0)
            score = self._explore_move(next_vehicle, metric, depth - 1)
            min_score = min(min_score, score)
        return min_score

    def _possible_moves(self, vehicle: Vehicle):
        if vehicle.boost_available:
            return self.moves
        return self.moves[1:]

    def _report_bad_prediction(self, vehicle: Vehicle, vehicle_id: int):
        prediction = self.predictions[vehicle_id]
        if prediction is None:
            return

        isBad = distance(prediction.position, vehicle.position) > 5
        isBad |= distance(prediction.speed, vehicle.speed) > 5
        isBad |= abs(mod_angle(prediction.direction) - mod_angle(vehicle.direction)) > 0.1
        if isBad:
            debug("BAD PREDICTION")
            debug("predicted:", prediction)
            debug("got:", vehicle)


"""
Inputs acquisition
"""


def read_checkpoint() -> Checkpoint:
    return np.array([int(j) for j in input().split()])


def read_checkpoints() -> List[Checkpoint]:
    checkpoint_count = int(input())
    return [read_checkpoint() for _ in range(checkpoint_count)]


def read_vehicle() -> Vehicle:
    x, y, vx, vy, angle, next_check_point_id = [int(j) for j in input().split()]
    return Vehicle(
        position=np.array([x, y]),
        speed=np.array([vx, vy]),
        direction=angle / 360 * 2 * math.pi,
        next_checkpoint_id=next_check_point_id,
        current_lap=0,
        boost_available=False)


"""
Game loop
"""


# TODO - add the SHIELD state + use (but first add the collisions)


def with_first_turn_orientation(track: Track, vehicle: Vehicle) -> Vehicle:
    # For the first turn, a pod can go in any direction, here we turn it toward the goal (forbids some moves though)
    return vehicle._replace(direction=track.angle_next_checkpoint(vehicle))


def game_loop():
    total_laps = int(input())
    checkpoints = read_checkpoints()
    track = Track(checkpoints, total_laps=total_laps)
    agent = ShortestPathAgent(track)

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


game_loop()
