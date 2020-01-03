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


Vector = np.ndarray


def vector(x: int, y: int) -> Vector:
    return np.array([x, y])


def distance2(v1: Vector, v2: Vector):
    v = v1 - v2
    return np.dot(v, v)


def distance(v1: Vector, v2: Vector):
    return math.sqrt(distance2(v1, v2))


"""
Input acquisition & Game constants
"""


WIDTH = 16001
HEIGHT = 7501


MAX_THRUST = 150
MAX_THROW_POWER = 500


class Goal(NamedTuple):
    x: int
    y_lo: int
    y_hi: int


LEFT_GOAL = Goal(x=0, y_lo=2150, y_hi=5500)
RIGHT_GOAL = Goal(x=WIDTH, y_lo=2150, y_hi=5500)
OWN_GOALS = (LEFT_GOAL, RIGHT_GOAL)


@dataclass()
class PlayerStatus:
    score: int
    magic: int


def read_status():
    score, magic = [int(i) for i in input().split()]
    return PlayerStatus(score=score, magic=magic)


# TODO - use a ECS system (would help move the objects)


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


@dataclass(frozen=True)
class GameState:
    player_goal: Goal
    opponent_goal: Goal
    player_wizards: List[Wizard]
    opponent_wizards: List[Wizard]
    snaffles: List[Snaffle]
    bludgers: List[Bludger]

    @classmethod
    def empty(cls, player_goal: Goal, opponent_goal: Goal):
        return cls(player_goal=player_goal, opponent_goal=opponent_goal,
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


class Action(NamedTuple):
    # Action for each wizard (0 ≤ thrust ≤ 150, 0 ≤ power ≤ 500)
    # i.e.: "MOVE x y thrust" or "THROW x y power"

    is_throw: bool  # Otherwise, it is a move
    direction: Vector
    power: int

    def __repr__(self):
        action_type = "THROW" if self.is_throw else "MOVE"
        x, y = self.direction
        return action_type + " " + str(x) + " " + str(y) + " " + str(self.power)


"""
------------------------------------------------------------------------------------------------------------------------
BASIC AGENTS
------------------------------------------------------------------------------------------------------------------------
"""


class Agent(abc.ABC):
    @abc.abstractmethod
    def get_actions(self, state: GameState) -> List[Action]:
        pass


class StupidAgent(Agent):
    def get_actions(self, state: GameState) -> List[Action]:
        return [Action(is_throw=False, direction=vector(8000, 3750), power=100),
                Action(is_throw=False, direction=vector(8000, 3750), power=100)]


class GrabClosestAndShootTowardGoal(Agent):
    """
    You need to keep a state here, in order to avoid the situation in which your agent changes
    direction suddenly and remains stuck, with two wizard each alternating snaffles
    """

    def __init__(self):
        self.targeted_snaffles = {}

    def get_actions(self, state: GameState) -> List[Action]:
        actions = []
        available_snaffles = list(state.snaffles)
        for wizard in state.player_wizards:
            if not wizard.has_snaffle:
                action = self._move_toward_snaffle(state, wizard, available_snaffles)
            else:
                # del self.targeted_snaffles[wizard.id]     # Commented to keep the same snaffle to the end
                action = self._shoot_toward_goal(state.opponent_goal)
            actions.append(action)
        return actions

    def _move_toward_snaffle(self, state, wizard, available_snaffles) -> Action:
        snaffle = None
        prefered_snaffle_id = self.targeted_snaffles.get(wizard.id)
        if prefered_snaffle_id:
            snaffle = self._find_by_id(available_snaffles, prefered_snaffle_id)
        if snaffle is None:
            snaffle = min(available_snaffles, key=lambda s: distance2(wizard.position, s.position), default=None)
            available_snaffles.remove(snaffle)
        if snaffle is None:
            # In case there is but one snaffle and it is already taken
            snaffle = state.snaffles[0]
        self.targeted_snaffles[wizard.id] = snaffle.id
        return Action(is_throw=False, direction=snaffle.position + snaffle.speed, power=MAX_THRUST)

    def _find_by_id(self, entities, identity):
        for entity in entities:
            if entity.id == identity:
                return entity
        return None

    def _shoot_toward_goal(self, goal):
        goal_center = vector(goal.x, int((goal.y_lo + goal.y_hi) / 2))
        return Action(is_throw=True, direction=goal_center, power=MAX_THROW_POWER)


# TODO - an evaluation function that counts the goal + tries to put the balls in the adversary camp


"""
------------------------------------------------------------------------------------------------------------------------
PHYSIC ENGINE
------------------------------------------------------------------------------------------------------------------------
"""


def apply_force(position: Vector, speed: Vector, thrust: float, direction: Vector) -> Vector:
    pass


def simulate(state: GameState, actions: Dict[int, Action]) -> GameState:
    pass


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
        game_state = read_state(player_goal, opponent_goal)

        debug("player status:", player_status)
        debug("opponent status:", opponent_status)
        debug("game state:", game_state)

        for action in agent.get_actions(game_state):
            print(action)


if __name__ == '__main__':
    game_loop(agent=GrabClosestAndShootTowardGoal())
