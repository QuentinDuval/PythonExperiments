from collections import namedtuple
import numpy as np
import sys
import math

"""
Constants
"""

WIDTH = 16000
HEIGHT = 9000

CHECKPOINT_RADIUS = 600
FORCE_FIELD_RADIUS = 400

MAX_TURN_DEG = 18
MAX_TURN_RAD = MAX_TURN_DEG / 360 * 2 * math.pi

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
Inputs
"""

Vector = np.ndarray

Vehicle = namedtuple('Vehicle', ['position', 'speed', 'direction'])

Checkpoint = namedtuple('Checkpoint', ['position'])

PlayerInput = namedtuple('PlayerInput', [
    'position',
    'checkpoint',
    'next_checkpoint_dist',
    'next_checkpoint_angle'
])

Opponent = namedtuple('Opponent', ['x', 'y'])

# Action = namedtuple('Action', ['dir_x', 'dir_y', 'thrust'])


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


def norm2(v):
    return np.dot(v, v)


def norm(vector):
    return math.sqrt(norm2(vector))


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
Model:
* when going in straight line: dv_dt = thrust - 0.15 * v
* TODO - max angle of rotation (18 degrees apparently: http://files.magusgeek.com/csb/csb.html)
"""


def get_vehicle(prev: PlayerInput, curr: PlayerInput) -> Vehicle:
    prev_position = prev.position
    curr_position = curr.position
    speed = curr_position - prev_position

    checkpoint_direction = player.checkpoint - player.position
    next_checkpoint_angle_rad = curr.next_checkpoint_angle / 180 * math.pi
    direction_angle = get_angle(checkpoint_direction)
    player_angle = direction_angle - next_checkpoint_angle_rad

    debug("direction angle:", direction_angle, checkpoint_direction)
    debug("next_cp angle:", next_checkpoint_angle_rad)
    debug("player angle:", player_angle)

    # player_angle = mod_angle(player_angle)
    return Vehicle(position=curr_position, speed=speed, direction=player_angle)


def next_position(vehicle: Vehicle, next_checkpoint: Vector, thrust: int, dt: float = 1.0) -> Vehicle:
    to_next_checkpoint = next_checkpoint - vehicle.position
    to_next_angle = get_angle(to_next_checkpoint)

    if to_next_angle > vehicle.direction:
        new_direction = min(vehicle.direction + MAX_TURN_RAD, to_next_angle)
    else:
        new_direction = max(vehicle.direction - MAX_TURN_RAD, to_next_angle)
    # new_direction = mod_angle(new_direction)

    dv_dt = np.array([
        thrust * math.cos(new_direction),
        thrust * math.sin(new_direction)
    ])

    new_speed = vehicle.speed * 0.85 + dv_dt  # the time step is one
    new_position = np.round(vehicle.position + new_speed * dt)
    return Vehicle(
        position=new_position,
        speed=np.trunc(new_speed),
        direction=new_direction)


"""
Agent

TODO:
* pour l'instant, l'agent rate la cible de temps à autre et doit faire un tour (overshoot)
* try to create a good model of the game: and compare the results
"""


# TODO - because we cannot see the future checkpoints, this AI that sees in the future is not great...


class MinimaxAgent:
    def __init__(self):
        self.boost_available = True
        self.previous_player: PlayerInput = None
        self.previous_opponent: Vector = None
        self.prediction: Vehicle = None

    def get_action(self, player: PlayerInput, opponent: Vector):
        if self.previous_player is None:
            self.previous_player = player
        if self.previous_opponent is None:
            self.previous_opponent = opponent

        thrust = self.search_action(player, opponent)
        if self.boost_available and thrust == 100 and player.next_checkpoint_dist > 3000:
            thrust = "BOOST"  # TODO - the boost should only be done if angle is perfect
            self.boost_available = False
        else:
            thrust = str(thrust)

        self.previous_player = player
        self.previous_opponent = opponent

        next_checkpoint_x, next_checkpoint_y = player.checkpoint
        return str(next_checkpoint_x) + " " + str(next_checkpoint_y) + " " + thrust

    def search_action(self, player: PlayerInput, opponent: Vector) -> str:
        vehicle = get_vehicle(self.previous_player, player)
        self.report_bad_prediction(vehicle)

        min_dist = float('inf')
        best_thrust = 0
        for thrust in [100, 0]:
            d = self.best_move(vehicle, player.checkpoint, thrust, depth=6)
            # debug("possible move:", thrust, d)
            if d < min_dist:
                min_dist = d
                best_thrust = thrust

        self.prediction = next_position(vehicle, player.checkpoint, best_thrust)
        debug("current:", vehicle)
        debug("prediction:", self.prediction)
        debug("move:", best_thrust, "for distance", min_dist)
        return best_thrust

    def report_bad_prediction(self, vehicle):
        if self.prediction is None:
            return

        isBad = distance(self.prediction.position, vehicle.position) > 5
        isBad |= distance(self.prediction.speed, vehicle.speed) > 5
        isBad |= abs(mod_angle(self.prediction.direction) - mod_angle(vehicle.direction)) > 0.1
        if isBad:
            debug("BAD PREDICTION")
            debug("predicted:", self.prediction)
            debug("got:", vehicle)

    def best_move(self, vehicle: Vehicle, next_checkpoint: Vector, thrust: int, depth: int) -> float:
        vehicle = next_position(vehicle, next_checkpoint, thrust)
        dist = distance2(vehicle.position, next_checkpoint)
        if depth == 0:
            return dist

        if dist < CHECKPOINT_RADIUS ** 2:
            return dist

        return min(self.best_move(vehicle, next_checkpoint, thrust, depth - 1) for thrust in [100, 0])


class IfAgent:
    def __init__(self):
        self.boost_available = True
        self.previous_player: PlayerInput = None
        self.previous_opponent: Vector = None

    def get_action(self, player: PlayerInput, opponent: Vector):
        self.previous_player = self.previous_player or player
        if self.previous_opponent is None:
            self.previous_opponent = opponent

        thrust = 0
        if -18 < player.next_checkpoint_angle < 18:
            thrust = 100
        elif -90 < player.next_checkpoint_angle < 90:
            if player.next_checkpoint_dist >= 300:
                thrust = 100
            elif player.next_checkpoint_dist >= 100:
                thrust = 30
            else:
                thrust = 10

        if self.boost_available and thrust == 100:
            thrust = "BOOST"
            self.boost_available = False
        else:
            thrust = str(thrust)

        self.previous_player = player
        self.previous_opponent = opponent
        next_checkpoint_x, next_checkpoint_y = player.checkpoint
        return str(next_checkpoint_x) + " " + str(next_checkpoint_y) + " " + thrust


"""
Game loop
"""

agent = MinimaxAgent()
while True:
    x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in
                                                                                               input().split()]
    opponent_x, opponent_y = [int(i) for i in input().split()]

    player = PlayerInput(
        position=np.array([x, y]),
        checkpoint=np.array([next_checkpoint_x, next_checkpoint_y]),
        next_checkpoint_dist=next_checkpoint_dist,
        next_checkpoint_angle=next_checkpoint_angle
    )
    opponent = np.array([opponent_x, opponent_y])

    debug(player)
    # debug("opponent:", opponent)
    action = agent.get_action(player, opponent)
    print(action)

