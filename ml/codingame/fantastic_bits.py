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
Utilities
"""


def debug(*args):
    print(*args, file=sys.stderr)


"""
Input acquisition & Game constants
"""


Vector = np.ndarray


def vector(x: int, y: int) -> Vector:
    return np.array([x, y])


WIDTH = 16001
HEIGHT = 7501


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


class Entity(NamedTuple):
    entity_id: int
    entity_type: str
    position: Vector
    speed: Vector
    has_snaffle: bool


def read_entities():
    entities = []
    entity_nb = int(input())
    for i in range(entity_nb):
        entity_id, entity_type, x, y, vx, vy, state = input().split()
        entity = Entity(
            entity_id=int(entity_id),
            entity_type=entity_type,
            position=np.array([int(x), int(y)]),
            speed=np.array([int(vx), int(vy)]),
            has_snaffle=int(state) > 0)
        entities.append(entity)
    return entities


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
AGENTS
------------------------------------------------------------------------------------------------------------------------
"""


class Agent(abc.ABC):
    @abc.abstractmethod
    def get_actions(self):
        pass


class StupidAgent(Agent):
    def get_actions(self):
        return [Action(is_throw=False, direction=vector(8000, 3750), power=100),
                Action(is_throw=False, direction=vector(8000, 3750), power=100)]



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
        entities = read_entities()

        debug("player status:", player_status)
        debug("opponent status:", opponent_status)
        debug("entities:", entities)

        for action in agent.get_actions():
            print(action)


if __name__ == '__main__':
    game_loop(agent=StupidAgent())
