import math
import sys
from collections import namedtuple
from typing import List, Tuple

import numpy as np


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


def get_angle(v):
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


def distance2(from_, to_):
    return norm2(to_ - from_)


def distance(from_, to_):
    return norm(to_ - from_)


def mod_angle(angle):
    if angle > 2 * math.pi:
        return angle - 2 * math.pi
    if angle < 0:
        return angle + 2 * math.pi
    return angle


"""
Data structures
"""

Vector = np.ndarray

Vehicle = namedtuple('Vehicle', [
    'position', 'speed', 'direction',
    'next_checkpoint_id', 'current_lap',
    'boost_available'])

Checkpoint = np.ndarray


"""
Movement
"""


class Track:
    def __init__(self, checkpoints: List[Checkpoint], total_laps: int):
        # Compute the distance to the end: you cannot just compute to next else IA will refuse to cross a checkpoint
        self.checkpoints = checkpoints
        self.total_checkpoints = checkpoints * total_laps
        self.distances = np.array([0] * len(self.total_checkpoints))
        for i in reversed(range(len(self.total_checkpoints) - 1)):
            self.distances[i] = self.distances[i + 1] + distance2(self.total_checkpoints[i],
                                                                  self.total_checkpoints[i + 1])

    # TODO - replace 'thrust' by 'action" (boost and shield), or consider that -1 is shield

    def __len__(self):
        return len(self.checkpoints)

    def remaining_distance2(self, vehicle: Vehicle) -> float:
        checkpoint_id = vehicle.next_checkpoint_id + vehicle.current_lap * len(self.checkpoints)
        return distance2(vehicle.position, self.total_checkpoints[checkpoint_id]) + self.distances[checkpoint_id]

    def next_checkpoint(self, vehicle: Vehicle) -> Checkpoint:
        checkpoint_id = vehicle.next_checkpoint_id + vehicle.current_lap * len(self.checkpoints)
        return self.total_checkpoints[checkpoint_id]

    def next_position(self, vehicle: Vehicle, thrust: int, diff_angle: float, dt: float = 1.0) -> Vehicle:
        new_direction = self._next_direction(vehicle, diff_angle)
        dv_dt = np.array([thrust * math.cos(new_direction), thrust * math.sin(new_direction)])
        new_speed = vehicle.speed + dv_dt  # the time step is one
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

    def _next_direction(self, vehicle: Vehicle, diff_angle: float) -> float:
        if diff_angle > 0:
            diff_angle = min(MAX_TURN_RAD, diff_angle)
        elif diff_angle < 0:
            diff_angle = max(-MAX_TURN_RAD, diff_angle)
        return mod_angle(vehicle.direction + diff_angle)

    def diff_angle_for_checkpoint(self, vehicle: Vehicle) -> float:
        # TODO - use the angle for the next checkpoint (good default when far from target)
        to_next_checkpoint = self.total_checkpoints[vehicle.next_checkpoint_id] - vehicle.position
        to_next_angle = get_angle(to_next_checkpoint)

        # Find the minimum angle between target and orientation
        diff_angle = to_next_angle - vehicle.direction
        if diff_angle > math.pi:
            diff_angle = diff_angle - 2 * math.pi
        elif diff_angle < -math.pi:
            diff_angle = diff_angle + 2 * math.pi

        return self._next_direction(vehicle, diff_angle)


"""
Game state
"""


class PlayerState:
    # TODO - you should track also the timer to next checkpoint... for you and the opponent

    def __init__(self):
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

    def complete_vehicle(self, vehicle: Vehicle, vehicle_id: int) -> Vehicle:
        return vehicle._replace(
            current_lap=self.laps[vehicle_id],
            boost_available=self.boost_available[vehicle_id]
        )


class GameState:
    def __init__(self):
        self.player = PlayerState()
        self.opponent = PlayerState()

    def track_lap(self, player: List[Vehicle], opponent: List[Vehicle]):
        self.player.track_lap(player)
        self.opponent.track_lap(opponent)


"""
Agent that just tries to minimize the distance, not taking into account collisions
"""


class ShortestPathAgent:
    def __init__(self, track: Track):
        self.track = track
        self.game_state = GameState()
        self.predictions: List[Vehicle] = [None, None]
        self.thursts = np.array([BOOST_STRENGTH, 100, 20])
        # TODO - there is a problem: if you put thrust=0, then the IA stay stuck in front of checkpoint sometimes

    def get_action(self, player: List[Vehicle], opponent: List[Vehicle]) -> List[str]:
        self.game_state.track_lap(player, opponent)
        actions = []
        for vehicle_id, vehicle in enumerate(player):
            vehicle = self.game_state.player.complete_vehicle(vehicle, vehicle_id)
            self._report_bad_prediction(vehicle, vehicle_id)
            action, next_vehicle = self._get_action(vehicle, vehicle_id)
            self.predictions[vehicle_id] = next_vehicle
            actions.append(action)
        return actions

    def _get_action(self, vehicle: Vehicle, vehicle_id: int) -> Tuple[str, Vehicle]:
        best_thrust = 0
        min_score = float('inf')
        best_next_vehicle = None
        for thrust in self._thursts(vehicle):
            next_vehicle = self.track.next_position(vehicle, thrust, self.track.diff_angle_for_checkpoint(vehicle))
            score = self._explore_move(next_vehicle, depth=6)

            debug("action:", thrust)
            debug("score:", score)

            if score < min_score:
                min_score = score
                best_thrust = thrust
                best_next_vehicle = next_vehicle

        debug("current:", vehicle)
        debug("best action:", best_thrust)
        debug("prediction:", best_next_vehicle)

        if best_thrust > 100:
            best_thrust = "BOOST"
            self.game_state.player.notify_boost_used(vehicle_id)
        else:
            best_thrust = str(best_thrust)

        cp_x, cp_y = self.track.next_checkpoint(vehicle)
        return str(cp_x) + " " + str(cp_y) + " " + best_thrust, best_next_vehicle

    def _explore_move(self, vehicle: Vehicle, depth: int) -> float:
        if depth == 0:
            return self.track.remaining_distance2(vehicle)

        diff_angle = self.track.diff_angle_for_checkpoint(vehicle)
        return min(self._explore_move(self.track.next_position(vehicle, thrust, diff_angle, dt=1.0), depth - 1)
                   for thrust in self._thursts(vehicle))

    def _thursts(self, vehicle: Vehicle):
        if vehicle.boost_available:
            return self.thursts
        return self.thursts[1:]

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


# TODO - for the first turn, a pod can go in any direction
# TODO - add the SHIELD state + use (but first add the collisions)


def game_loop():
    total_laps = int(input())
    checkpoints = read_checkpoints()
    track = Track(checkpoints, total_laps=total_laps)
    agent = ShortestPathAgent(track)

    debug("laps", total_laps)
    debug("checkpoints:", checkpoints)

    while True:
        player_vehicles = [read_vehicle() for _ in range(2)]
        opponent_vehicles = [read_vehicle() for _ in range(2)]
        actions = agent.get_action(player_vehicles, opponent_vehicles)
        for action in actions:
            print(action)


game_loop()
