"""
Grab Snaffles and try to throw them through the opponent's goal!
Move towards a Snaffle and use your team id to determine where you need to throw it.
"""


import abc
from collections import *
from dataclasses import *
import numpy as np
from typing import *
import sys
import math


"""
Utilities for debugging an Geometry
"""


def debug(*args):
    print(*args, file=sys.stderr)


"""
Geometry
"""


Angle = float
Mass = float
Vector = np.ndarray


def vector(x: int, y: int) -> Vector:
    return np.array([x, y])


def norm2(v) -> float:
    return np.dot(v, v)


def norm(v) -> float:
    return math.sqrt(norm2(v))


def distance2(v1: Vector, v2: Vector):
    v = v1 - v2
    return np.dot(v, v)


def distance(v1: Vector, v2: Vector):
    return math.sqrt(distance2(v1, v2))


def get_angle(v: Vector) -> Angle:
    # Get angle from a vector (x, y) in radian
    x, y = v
    if x > 0:
        return np.arctan(y / x)
    if x < 0:
        return math.pi - np.arctan(- y / x)
    return math.pi / 2 if y >= 0 else -math.pi / 2


def mod_angle(angle: Angle) -> Angle:
    if angle > 2 * math.pi:
        return angle - 2 * math.pi
    if angle < 0:
        return angle + 2 * math.pi
    return angle


"""
Input acquisition & Game constants
"""


WIDTH = 16001
HEIGHT = 7501

MAX_THRUST = 150
MAX_THROW_POWER = 500

WIZARD_RADIUS = 400
BLUDGER_RADIUS = 200
SNAFFLE_RADIUS = 150


class Goal(NamedTuple):
    x: int
    y_lo: int
    y_hi: int

    def center(self):
        return vector(self.x, (self.y_li + self.y_lo) // 2)


LEFT_GOAL = Goal(x=0, y_lo=2150, y_hi=5500)
RIGHT_GOAL = Goal(x=WIDTH, y_lo=2150, y_hi=5500)
OWN_GOALS = (LEFT_GOAL, RIGHT_GOAL)


@dataclass(frozen=False)
class PlayerStatus:
    score: int
    magic: int


def read_status():
    score, magic = [int(i) for i in input().split()]
    return PlayerStatus(score=score, magic=magic)


class Wizard(NamedTuple):
    id: int
    position: Vector
    speed: Vector
    has_snaffle: bool


class Snaffle(NamedTuple):
    id: int
    position: Vector
    speed: Vector


class Bludger(NamedTuple):
    id: int
    position: Vector
    speed: Vector


"""
------------------------------------------------------------------------------------------------------------------------
GAME STATE
------------------------------------------------------------------------------------------------------------------------
"""


@dataclass(frozen=True)
class GameState:
    player_status: PlayerStatus
    opponent_status: PlayerStatus
    player_goal: Goal
    opponent_goal: Goal
    player_wizards: List[Wizard]
    opponent_wizards: List[Wizard]
    snaffles: List[Snaffle]
    bludgers: List[Bludger]

    @classmethod
    def empty(cls, player_status: PlayerStatus, opponent_status: PlayerStatus, player_goal: Goal, opponent_goal: Goal):
        return cls(player_status=player_status, opponent_status=opponent_status,
                   player_goal=player_goal, opponent_goal=opponent_goal,
                   player_wizards=[], opponent_wizards=[],
                   snaffles=[], bludgers=[])

    def init_next(self):
        return GameState(
            player_status=self.player_status, opponent_status=self.opponent_status,
            player_goal=self.player_goal, opponent_goal=self.opponent_goal,
            player_wizards=[], opponent_wizards=[],
            snaffles=[], bludgers=[])


def read_state(player_goal: Goal, opponent_goal: Goal) -> GameState:
    game_state = GameState.empty(player_goal, opponent_goal)
    entity_nb = int(input())
    for _ in range(entity_nb):
        entity_id, entity_type, x, y, vx, vy, state = input().split()
        entity_id = int(entity_id)
        position = vector(int(x), int(y))
        speed = vector(int(vx), int(vy))
        has_snaffle = int(state) > 0
        if entity_type == "WIZARD":
            game_state.player_wizards.append(Wizard(id=entity_id, position=position,
                                                    speed=speed, has_snaffle=has_snaffle))
        elif entity_type == "OPPONENT_WIZARD":
            game_state.opponent_wizards.append(Wizard(id=entity_id, position=position,
                                                      speed=speed, has_snaffle=has_snaffle))
        elif entity_type == "BLUDGER":
            game_state.bludgers.append(Bludger(id=entity_id, position=position, speed=speed))
        else:
            game_state.snaffles.append(Snaffle(id=entity_id, position=position, speed=speed))
    return game_state


"""
------------------------------------------------------------------------------------------------------------------------
ACTIONS
------------------------------------------------------------------------------------------------------------------------
"""


class Move(NamedTuple):
    is_throw: bool  # Otherwise, it is a move
    direction: Vector
    power: int

    def __repr__(self):
        action_type = "THROW" if self.is_throw else "MOVE"
        x, y = self.direction
        return action_type + " " + str(x) + " " + str(y) + " " + str(self.power)


class Spell(NamedTuple):
    name: str
    target_id: int

    def __repr__(self):
        return self.name + " " + str(self.target_id)


Action = Union[Move, Spell]


"""
------------------------------------------------------------------------------------------------------------------------
PHYSIC ENGINE
------------------------------------------------------------------------------------------------------------------------
"""


T = TypeVar('T')


def apply_force(entity: T, thrust: float, destination: Vector, friction: float, mass: Mass, dt=1.0) -> T:
    direction = get_angle(destination - entity.position)
    force = np.array([thrust * math.cos(direction), thrust * math.sin(direction)])
    dv_dt = force / mass
    new_speed = entity.speed + dv_dt * dt
    new_position = np.round(entity.position + new_speed * dt)
    # TODO - take into account the borders ? beware of snaffles
    return entity._replace(
        position=new_position,
        speed=np.trunc(new_speed * friction))


def intersect_goal(snaffle: Snaffle, next_snaffle: Snaffle, goal: Goal):
    pass


def simulate(state: GameState, actions: List[Tuple[Wizard, Action]]) -> GameState:
    next_state = GameState.init_next()

    # Move the snaffle
    for snaffle in state.snaffles:
        thrust = 0.
        destination = state.opponent_goal.center()
        for wizard, action in actions:
            if wizard.position == snaffle.position:
                if isinstance(action, Move) and action.is_throw:
                    thrust = action.power   # TODO - inherit the speed of the player?
                    destination = action.direction
                    break
        next_snaffle = apply_force(snaffle, thrust=thrust, destination=destination, friction=0.75, mass=0.5, dt=1.0)
        if intersect_goal(snaffle, next_snaffle, state.opponent_goal):
            next_state.player_status.score += 1
        elif intersect_goal(snaffle, next_snaffle, state.player_goal):
            next_state.opponent_status.score += 1
        else:
            next_state.snaffles.append(next_snaffle)

    # Move the wizard
    for wizard in state.player_wizards:
        thrust = 0.
        destination = state.opponent_goal.center()
        for action_wizard, action in actions:
            if action_wizard.id == wizard.id:
                if isinstance(action, Move) and not action.is_throw:
                    thrust = action.power
                    destination = action.direction
                    break
        next_wizard = apply_force(wizard, thrust=thrust, destination=destination, friction=0.75, mass=1.0, dt=1.0)
        next_state.player_wizards.append(next_wizard)

    # Move the opponent wizard: TODO - move them according to a basic AI
    for wizard in state.opponent_wizards:
        next_wizard = apply_force(wizard, thrust=0., destination=state.player_goal.center(), friction=0.75, mass=1.0, dt=1.0)
        next_state.opponent_wizards.append(next_wizard)

    # Move the bludgers: TODO - how does the acceleration of bludgers work?
    for bludger in state.bludgers:
        next_bludger = apply_force(bludger, thrust=0., destination=bludger.position, friction=0.9, mass=8.0, dt=1.0)
        state.bludgers.append(next_bludger)

    # Increase the magic
    next_state.player_status.magic += 1
    next_state.opponent_status.magic += 1
    return next_state


"""
------------------------------------------------------------------------------------------------------------------------
BASIC AGENTS
------------------------------------------------------------------------------------------------------------------------
"""


class Agent(abc.ABC):
    @abc.abstractmethod
    def get_actions(self, state: GameState) -> List[Move]:
        pass


class StupidAgent(Agent):
    def get_actions(self, state: GameState) -> List[Move]:
        return [Move(is_throw=False, direction=vector(8000, 3750), power=100),
                Move(is_throw=False, direction=vector(8000, 3750), power=100)]


class GrabClosestAndShootTowardGoal(Agent):
    """
    You need to keep a state here, in order to avoid the situation in which your agent changes
    direction suddenly and remains stuck, with two wizard each alternating snaffles
    """

    def __init__(self):
        self.wizard_snaffles = {}

    def get_actions(self, state: GameState) -> List[Action]:
        actions = []

        targeted_snaffles = set(self.wizard_snaffles.values())
        opponent_snaffles = self._opponent_snaffles(state)
        available_snaffles = [s for s in state.snaffles
                              if s.id not in targeted_snaffles and s.id not in opponent_snaffles]

        # TODO - specialize the wizard: one to defend, one to attack
        for wizard in state.player_wizards:
            bludger = self._incoming_bludger(state, wizard)
            # TODO - if a snaffle is too far away, and moving toward you camp, take it
            if bludger is not None and state.player_status.magic >= 5:
                action = Spell(name="OBLIVIATE", target_id=bludger.id)
            elif not wizard.has_snaffle:
                action = self._move_toward_snaffle(state, wizard, available_snaffles)
            else:
                # del self.targeted_snaffles[wizard.id]     # Commented to keep the same snaffle to the end
                action = self._shoot_toward_goal(state, wizard, state.opponent_goal)
            actions.append(action)

        return actions

    def _opponent_snaffles(self, state: GameState) -> Set[Snaffle]:
        taken = set()
        for wizard in state.opponent_wizards:
            if wizard.has_snaffle:
                for snaffle in state.snaffles:
                    if distance2(snaffle.position, wizard.position) < WIZARD_RADIUS ** 2:
                        taken.add(snaffle.id)
                        break
        return taken

    def _incoming_bludger(self, state: GameState, wizard: Wizard) -> Bludger:
        # TODO - or check the cosine similarity + distance ?
        next_positions = [b.position + b.speed * 4 for b in state.bludgers]
        for i, next_position in enumerate(next_positions):
            if distance(wizard.position, next_position) < WIZARD_RADIUS + BLUDGER_RADIUS:
                return state.bludgers[i]
        return None

    def _move_toward_snaffle(self, state, wizard, available_snaffles) -> Move:
        snaffle = None
        prefered_snaffle_id = self.wizard_snaffles.get(wizard.id)
        if prefered_snaffle_id:
            snaffle = self._find_by_id(state.snaffles, prefered_snaffle_id)
        if snaffle is None:
            snaffle = min(available_snaffles, key=lambda s: distance2(wizard.position, s.position), default=None)
            if snaffle is not None:
                available_snaffles.remove(snaffle)
        if snaffle is None:
            # In case there is but one snaffle and it is already taken
            snaffle = state.snaffles[0]
        self.wizard_snaffles[wizard.id] = snaffle.id
        return Move(is_throw=False, direction=snaffle.position + snaffle.speed, power=MAX_THRUST)

    def _find_by_id(self, entities, identity):
        for entity in entities:
            if entity.id == identity:
                return entity
        return None

    def _shoot_toward_goal(self, state, wizard, goal):
        # TODO - avoid to shoot toward an opponent wizard OR toward a bludger
        return Move(is_throw=True, direction=goal.center() - wizard.speed, power=MAX_THROW_POWER)


# TODO - an evaluation function that counts the goal + tries to put the balls in the adversary camp


"""
------------------------------------------------------------------------------------------------------------------------
GAME LOOP
------------------------------------------------------------------------------------------------------------------------
"""


def game_loop(agent: Agent):
    # if 0 you need to score on the right of the map, if 1 you need to score on the left
    my_team_id = int(input())
    player_goal = OWN_GOALS[my_team_id]
    opponent_goal = OWN_GOALS[1-my_team_id]

    debug("my team id:", my_team_id)
    debug("my goal:", player_goal)
    debug("target goal:", opponent_goal)

    while True:
        player_status = read_status()
        opponent_status = read_status()
        game_state = read_state(player_status, opponent_status, player_goal, opponent_goal)

        debug("player status:", player_status)
        debug("opponent status:", opponent_status)
        debug("game state:", game_state)

        for action in agent.get_actions(game_state):
            print(action)


if __name__ == '__main__':
    game_loop(agent=GrabClosestAndShootTowardGoal())
