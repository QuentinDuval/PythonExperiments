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

Vector = namedtuple('Position', ['x', 'y'])

Vehicle = namedtuple('Vehicle', ['position', 'speed', 'direction'])

PlayerInput = namedtuple('PlayerInput', [
    'x', 'y',
    'next_checkpoint_x', 'next_checkpoint_y',
    'next_checkpoint_dist', 'next_checkpoint_angle'
])

Opponent = namedtuple('Opponent', ['x', 'y'])

# Action = namedtuple('Action', ['dir_x', 'dir_y', 'thrust'])


"""
Vector arithmetic
"""


def get_angle(vector: Vector):
    # Get angle from a vector (x, y)
    if vector.x > 0:
        return np.arctan(vector.y / vector.x)
    if vector.x < 0:
        return math.pi - np.arctan(- vector.y / vector.x)
    return math.pi / 2 if vector.y >= 0 else -math.pi / 2


def add_vector(v1: Vector, v2: Vector):
    return Vector(x=v1.x + v2.x, y=v1.y + v2.y)


def sub_vector(v1: Vector, v2: Vector):
    return Vector(x=v1.x - v2.x, y=v1.y - v2.y)


def norm2(v: Vector):
    return v.x ** 2 + v.y ** 2


def norm(vector):
    return math.sqrt(norm2(vector))


def distance2(x, y):
    return norm2(sub_vector(x, y))


def distance(x, y):
    return norm(sub_vector(x, y))


"""
Model:
* when going in straight line: dv_dt = thrust - 0.15 * v
* TODO - max angle of rotation (18 degrees apparently: http://files.magusgeek.com/csb/csb.html)
"""


def position_vector(player: PlayerInput) -> Vector:
    return Vector(x=player.x, y=player.y)


def checkpoint_vector(player: PlayerInput) -> Vector:
    return Vector(x=player.next_checkpoint_x, y=player.next_checkpoint_y)


def direction_vector(player: PlayerInput) -> Vector:
    return Vector(x=player.next_checkpoint_x - player.x, y=player.next_checkpoint_y - player.y)


def get_vehicle(prev: PlayerInput, curr: PlayerInput) -> Vehicle:
    prev_position = position_vector(prev)
    position = position_vector(curr)
    speed = norm(sub_vector(position, prev_position))

    checkpoint_direction = direction_vector(curr)
    direction_angle = get_angle(checkpoint_direction)
    player_angle = direction_angle - curr.next_checkpoint_angle
    return Vehicle(position=position, speed=speed, direction=player_angle)


def next_position(vehicle: Vehicle, next_checkpoint: Vector, thrust: int) -> Vehicle:
    speed_norm = vehicle.speed
    speed = Vector(
        x=speed_norm * math.cos(vehicle.direction),
        y=speed_norm * math.sin(vehicle.direction))

    to_next_checkpoint = sub_vector(vehicle.position, next_checkpoint)
    to_next_angle = get_angle(to_next_checkpoint)

    if to_next_angle > vehicle.direction:
        new_direction = min(vehicle.direction + 18, to_next_angle)
    else:
        new_direction = max(vehicle.direction - 18, to_next_angle)

    dv_dt = Vector(
        x=thrust * math.cos(new_direction),
        y=thrust * math.sin(new_direction))

    new_speed = add_vector(speed, dv_dt)  # the time step is one
    return Vehicle(position=add_vector(vehicle.position, new_speed), speed=norm(new_speed), direction=new_direction)


"""
Agent

TODO:
* pour l'instant, l'agent rate la cible de temps à autre et doit faire un tour (overshoot)
* try to create a good model of the game: and compare the results
"""


# TODO - because we cannot see the future checkpoints, this AI that sees in the future is not great...


class TryAgent:
    def __init__(self):
        self.boost_available = True
        self.previous_player: PlayerInput = None
        self.previous_opponent: Vector = None

    def get_action(self, player: PlayerInput, opponent: Vector):
        self.previous_player = self.previous_player or player
        self.previous_opponent = self.previous_opponent or opponent

        if -18 < player.next_checkpoint_angle < 18 and player.next_checkpoint_dist > 1000:
            thrust = 100
        elif player.next_checkpoint_dist > 3000 and -90 < player.next_checkpoint_angle < 90:
            thrust = 100
        elif self.opponent_in_front(player, opponent):
            thrust = 100
        else:
            thrust = self.search_action(player, opponent)

        if self.boost_available and thrust == 100 and player.next_checkpoint_dist > 3000:
            thrust = "BOOST"
            self.boost_available = False
        else:
            thrust = str(thrust)
        self.previous_player = player
        self.previous_opponent = opponent
        return str(player.next_checkpoint_x) + " " + str(player.next_checkpoint_y) + " " + thrust

    def opponent_in_front(self, player: PlayerInput, opponent: Vector) -> bool:
        # TODO - find a better way to hit the opponent
        vehicle = get_vehicle(self.previous_player, player)
        next_checkpoint = checkpoint_vector(player)
        d1 = distance2(vehicle.position, next_checkpoint)
        d2 = distance2(opponent, next_checkpoint)
        if d1 > d2:
            return False

        diff_opp = sub_vector(opponent, self.previous_opponent)
        next_opp = add_vector(opponent, diff_opp)
        vehicle = next_position(vehicle, next_checkpoint, 100)

        to_opponent = distance2(vehicle.position, next_opp)
        if to_opponent < FORCE_FIELD_RADIUS ** 2:
            return True
        return False

    def search_action(self, player: PlayerInput, opponent: Vector) -> str:
        # TODO - it does not see very far in the future and tend to be too careful when near a checkpoint or past a checkpoint

        vehicle = get_vehicle(self.previous_player, player)
        next_checkpoint = checkpoint_vector(player)

        min_dist = float('inf')
        best_thrust = 0
        for thrust in [100, 40]:
            d = self.best_move(vehicle, next_checkpoint, thrust, depth=6)
            debug(d, thrust)
            if d < min_dist:
                min_dist = d
                best_thrust = thrust
        return best_thrust

    def best_move(self, vehicle: Vehicle, next_checkpoint: Vector, thrust: int, depth: int) -> float:
        vehicle = next_position(vehicle, next_checkpoint, thrust)
        dist = distance2(vehicle.position, next_checkpoint)
        if depth == 0:
            return dist

        if dist < CHECKPOINT_RADIUS ** 2:
            return dist

        return min(self.best_move(vehicle, next_checkpoint, thrust, depth - 1) for thrust in [100, 40])


class IfAgent:
    def __init__(self):
        self.boost_available = True
        # self.traces = []

    def get_action(self, player: PlayerInput, opponent: Vector):
        thrust = 0
        if -18 < player.next_checkpoint_angle < 18:
            thrust = 100
        elif -45 < player.next_checkpoint_angle < 45:
            thrust = 70
        elif -90 < player.next_checkpoint_angle < 90:
            if player.next_checkpoint_dist >= 200:
                thrust = 100
            else:
                thrust = 20

        if self.boost_available and thrust == 100:
            thrust = "BOOST"
            self.boost_available = False
        else:
            thrust = str(thrust)
        return str(player.next_checkpoint_x) + " " + str(player.next_checkpoint_y) + " " + thrust


"""
Game loop
"""

agent = TryAgent()
while True:
    player = PlayerInput(*[int(i) for i in input().split()])
    opponent = Vector(*[int(i) for i in input().split()])
    debug(player)
    action = agent.get_action(player, opponent)
    print(action)

